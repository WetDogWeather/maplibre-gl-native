#include <mbgl/util/thread_pool.hpp>

#include <mbgl/platform/settings.hpp>
#include <mbgl/platform/thread.hpp>
#include <mbgl/util/instrumentation.hpp>
#include <mbgl/util/monotonic_timer.hpp>
#include <mbgl/util/platform.hpp>
#include <mbgl/util/string.hpp>

namespace mbgl {

#ifndef NDEBUG
namespace {
constexpr auto MaxTasks = 1u << 20;
}
#endif

ThreadedSchedulerBase::~ThreadedSchedulerBase() = default;

void ThreadedSchedulerBase::terminate() {
    {
        std::lock_guard lock{workerMutex};
        terminated = true;
    }

    // Wake up all threads so that they shut down
    cvAvailable.notify_all();
}

std::thread ThreadedSchedulerBase::makeSchedulerThread(const size_t threadIndex) {
    return std::thread([this, threadIndex] {
        auto& settings = platform::Settings::getInstance();
        auto value = settings.get(platform::EXPERIMENTAL_THREAD_PRIORITY_WORKER);
        if (auto* priority = value.getDouble()) {
            platform::setCurrentThreadPriority(*priority);
        }

        const auto name = schedulerName + util::toString(threadIndex + 1);
        platform::setCurrentThreadName(name);
        MLN_TRACE_THREAD_NAME_HINT_STR(name, uniqueID.id());
        platform::attachThread();

        owningThreadPool.set(this);

        const auto threadQueueIndex = queueIndexFor(threadIndex);
        const auto generalQueueIndex = queueIndexFor({});

        auto& generalTaskCount = taskCounts[generalQueueIndex];
        auto& threadTaskCount = taskCounts[threadQueueIndex];

        std::vector<std::shared_ptr<Queue>> pending;

        while (true) {
            {
                MLN_TRACE_ZONE(idle); // waiting for something to do
                std::unique_lock conditionLock{workerMutex};

                // Wait for things this thread can do, or for a notification to shut down
                cvAvailable.wait(conditionLock, [&] { return terminated || generalTaskCount || threadTaskCount; });

                if (terminated) {
                    platform::detachThread();
                    break;
                }
            }

            // 1. Gather buckets for us to visit this iteration
            {
                pending.clear();
                std::lock_guard lock{taggedQueueMutex};
                std::ranges::transform(taggedQueue, std::back_inserter(pending), [](auto& kv) { return kv.second; });
            }

            // 2. Visit a task from each
            for (auto& q : pending) {
                std::function<void()> tasklet;
                {
                    MLN_TRACE_ZONE(pop);
                    std::lock_guard lock{q->mutex};
                    auto& generalQueue = q->queues[generalQueueIndex];
                    auto& threadQueue = q->queues[threadQueueIndex];

                    if (!threadQueue.empty()) {
                        // There's a thread-specific task pending
                        tasklet = std::move(threadQueue.front());
                        threadQueue.pop();
                        assert(tasklet);
                        --threadTaskCount;
                        [[maybe_unused]] const auto newCount = threadTaskCount.load();
                        assert(newCount < MaxTasks);
                    } else if (!generalQueue.empty()) {
                        // There's a generic task pending
                        tasklet = std::move(generalQueue.front());
                        generalQueue.pop();
                        assert(tasklet);
                        [[maybe_unused]] const auto newCount = --generalTaskCount;
                        assert(newCount < MaxTasks);
                    }
                    if (tasklet) {
                        ++q->runningCount;
                    } else {
                        // Nothing to do for this queue
                        continue;
                    }
                }

                try {
                    {
                        MLN_TRACE_ZONE(task);
                        tasklet();
                    }
                    {
                        MLN_TRACE_ZONE(cleanup);
                        // destroy the function and release its captures before unblocking `waitForEmpty`
                        tasklet = {};
                    }

                    // If this is the last thing running for this queue, signal any waiting `waitForEmpty`
                    if (!--q->runningCount) {
                        std::lock_guard lock{q->mutex};
                        if (q->empty()) {
                            q->cv.notify_all();
                        }
                    }
                } catch (...) {
                    std::lock_guard lock{q->mutex};
                    if (handler) {
                        handler(std::current_exception());
                    }

                    tasklet = {};

                    if (!--q->runningCount && q->empty()) {
                        q->cv.notify_all();
                    }

                    if (handler) {
                        continue;
                    }
                    throw;
                }
            }
        }
    });
}

void ThreadedSchedulerBase::schedule(std::function<void()>&& fn) {
    schedule({}, uniqueID, std::move(fn));
}

void ThreadedSchedulerBase::schedule(const util::SimpleIdentity tag, std::function<void()>&& fn) {
    schedule({}, tag, std::move(fn));
}

void ThreadedSchedulerBase::schedule(std::optional<std::size_t> threadIndex,
                                     util::SimpleIdentity tag,
                                     std::function<void()>&& fn) {
    MLN_TRACE_FUNC();
    assert(fn);
    if (!fn) return;

    tag = tag.isEmpty() ? uniqueID : tag;

    const auto queueIndex = queueIndexFor(threadIndex);

    std::shared_ptr<Queue> q;
    {
        MLN_TRACE_ZONE(queue);
        std::lock_guard lock{taggedQueueMutex};

        // find a matching bucket or insert a new entry
        auto result = taggedQueue.insert(std::make_pair(tag, std::shared_ptr<Queue>{}));
        if (result.second) {
            // new entry inserted, create the bucket for it
            result.first->second = std::make_shared<Queue>(threadCount + 1);
#ifdef MLN_TRACY_ENABLE
            const auto lockName = schedulerName + " queue " + util::toString(tag);
            MLN_LOCK_NAME_STR(result.first->second->mutex, lockName);
#endif
        }
        q = result.first->second;
    }

    {
        MLN_TRACE_ZONE(push);
        std::lock_guard lock{q->mutex};
        q->queues[queueIndex].push(std::move(fn));

        [[maybe_unused]] const auto newCount = ++taskCounts[queueIndex];
        assert(newCount > 0);
        MLN_ZONE_VALUE(newCount);
    }

    // Take the worker lock before notifying to prevent threads from waiting while we try to wake them
    {
        std::lock_guard workerLock{workerMutex};
        if (threadIndex) {
            cvAvailable.notify_all();
        } else {
            cvAvailable.notify_one();
        }
    }
}

void ThreadedSchedulerBase::waitForEmpty(const util::SimpleIdentity tag) {
    // Must not be called from a thread in our pool, or we would deadlock
    assert(!thisThreadIsOwned());
    if (!thisThreadIsOwned()) {
        const auto tagToFind = tag.isEmpty() ? uniqueID : tag;

        // Find the relevant bucket
        std::shared_ptr<Queue> q;
        {
            std::lock_guard lock{taggedQueueMutex};
            if (const auto it = taggedQueue.find(tagToFind); it != taggedQueue.end()) {
                q = it->second;
            } else {
                // Missing, probably already waited-for and removed
                return;
            }
        }

        {
            std::unique_lock queueLock{q->mutex};
            while (!q->empty() || q->runningCount) {
                q->cv.wait(queueLock);
            }
        }

        // After waiting for the queue to empty, go ahead and erase it from the map.
        {
            std::lock_guard lock{taggedQueueMutex};
            taggedQueue.erase(tagToFind);
        }
    }
}

} // namespace mbgl
