#include <mbgl/util/thread_pool.hpp>

#include <mbgl/platform/settings.hpp>
#include <mbgl/platform/thread.hpp>
#include <mbgl/util/instrumentation.hpp>
#include <mbgl/util/monotonic_timer.hpp>
#include <mbgl/util/platform.hpp>
#include <mbgl/util/string.hpp>

namespace mbgl {

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

        const auto name = schedulerName + util::toString(uniqueID) + " " + util::toString(threadIndex + 1);
        platform::setCurrentThreadName(name);
        MLN_TRACE_THREAD_NAME_HINT_STR(name, uniqueID.id());
        platform::attachThread();

        owningThreadPool.set(this);

        std::vector<std::shared_ptr<Queue>> pending;

        while (true) {
            {
                MLN_TRACE_ZONE(idle); // waiting for something to do
                std::unique_lock conditionLock{workerMutex};

                // Wait for things this thread can do, or for a notification to shut down
                cvAvailable.wait(conditionLock, [&] { return terminated || taskCount; });

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

                    if (!q->empty()) {
                        // There's a thread-specific task pending
                        tasklet = std::move(q->queue.front());
                        q->queue.pop_front();
                        assert(tasklet);

                        std::unique_lock workerLock{workerMutex};
                        assert(0 < taskCount);
                        --taskCount;
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
    schedule(uniqueID, std::move(fn));
}

void ThreadedSchedulerBase::schedule(const util::SimpleIdentity tag, std::function<void()>&& fn) {
    schedule(tag, &fn, 1);
}

void ThreadedSchedulerBase::schedule(const util::SimpleIdentity tag, std::vector<std::function<void()>>&& fs) {
    schedule(tag, fs.data(), fs.size());
    fs.clear();
}

void ThreadedSchedulerBase::schedule(util::SimpleIdentity tag, std::function<void()>* const functions, const std::size_t functionCount) {
    MLN_TRACE_FUNC();
    assert(std::all_of(functions, functions+functionCount, [](const auto& x) { return !!x; }));

    // Use the scheduler's tag if none is specified
    tag = tag.isEmpty() ? uniqueID : tag;

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
            const auto lockName = schedulerName + util::toString(uniqueID) + " queue" + util::toString(tag);
            MLN_LOCK_NAME_STR(result.first->second->mutex, lockName);
#endif
        }
        q = result.first->second;
    }

    // Increment the task count before adding tasks, or it can briefly go negative.
    // This is done within the mutex as required by the condition variable.
    {
        std::lock_guard workerLock{workerMutex};
        taskCount += functionCount;
        assert(taskCount >= functionCount);
        MLN_ZONE_VALUE(taskCount);
    }

    {
        MLN_TRACE_ZONE(push);
        std::lock_guard lock{q->mutex};
        if (q->closed) {
            assert(false);  // Don't add tasks to a queue while waiting for it to become empty
            return;
        }
        std::move(functions, functions+functionCount, std::back_inserter(q->queue));
    }

    // Wake up one more more threads to handle the new task(s)
    // We don't need to hold `workerMutex` because all modifications to conditions are made within it
    {
        MLN_TRACE_ZONE(notify);
        if (functionCount > 1) {
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
            q->closed = true;
            q->cv.wait(queueLock, [&]{ return (q->empty() && q->runningCount == 0); });
        }

        // After waiting for the queue to empty, go ahead and erase it from the map.
        {
            std::lock_guard lock{taggedQueueMutex};
            assert(q->empty());
            taggedQueue.erase(tagToFind);
        }
    }
}

} // namespace mbgl
