#pragma once

#include <mbgl/actor/mailbox.hpp>
#include <mbgl/actor/scheduler.hpp>
#include <mbgl/util/thread_local.hpp>
#include <mbgl/util/containers.hpp>
#include <mbgl/util/identity.hpp>
#include <mbgl/util/instrumentation.hpp>

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <ranges>
#include <thread>
#include <vector>

namespace mbgl {

class ThreadedSchedulerBase : public Scheduler {
public:
    /// @brief Schedule a generic task not assigned to any particular owner.
    /// The scheduler itself will own the task.
    /// @param fn Task to run
    void schedule(std::function<void()>&& fn) override;

    /// @brief Schedule a task assigned to the given owner `tag`.
    /// @param tag Identifier object to indicate ownership of `fn`, no tag indicates that the task is owned by the
    /// scheduler.
    /// @param fn Task to run
    void schedule(util::SimpleIdentity tag, std::function<void()>&& fn) override;

    // schedule multiple (with a single notify)
    void schedule(util::SimpleIdentity, std::vector<std::function<void()>>&&) override;

    /// @brief Set the prefix used for thread names
    void setName(std::string str) { schedulerName = std::move(str); }

    const util::SimpleIdentity uniqueID;
    const std::size_t threadCount;

protected:
    ThreadedSchedulerBase(std::size_t threadCount_, std::string name_)
        : threadCount(threadCount_),
          schedulerName(std::move(name_)) {
#ifdef MLN_TRACY_ENABLE
        auto lockName = schedulerName + util::toString(uniqueID) + " worker";
        MLN_LOCK_NAME_STR(workerMutex, lockName);
        lockName = schedulerName + util::toString(uniqueID) + " tagq";
        MLN_LOCK_NAME_STR(taggedQueueMutex, lockName);
#endif
    }
    ~ThreadedSchedulerBase() override;

    void terminate();
    std::thread makeSchedulerThread(size_t index);

    void schedule(util::SimpleIdentity, std::function<void()>*, std::size_t);

    /// @brief Wait until there's nothing pending or in process
    /// Must not be called from a task provided to this scheduler.
    /// @param tag Tag of the owner to identify the collection of tasks to
    ///            wait for. Not providing a tag waits on tasks owned by the scheduler.
    void waitForEmpty(const util::SimpleIdentity = util::SimpleIdentity::Empty) override;

    /// Returns true if called from a thread managed by the scheduler
    bool thisThreadIsOwned() const { return owningThreadPool.get() == this; }

    // Signal when an item is added to the queue
    MLN_TRACE_CONDITION_VAR cvAvailable;
    MLN_TRACE_LOCKABLE(std::mutex, workerMutex);
    MLN_TRACE_LOCKABLE(std::mutex, taggedQueueMutex);
    util::ThreadLocal<ThreadedSchedulerBase> owningThreadPool;
    std::uint32_t taskCount{0};
    std::string schedulerName;
    bool terminated{false};

    using TaskQueue = std::deque<std::function<void()>>;

    // Task queues bucketed by tag address
    struct Queue {
        std::atomic<std::size_t> runningCount; /* running tasks */
        MLN_TRACE_CONDITION_VAR cv;            /* queue empty condition */
        MLN_TRACE_LOCKABLE(std::mutex, mutex); /* lock */
        TaskQueue queue;                       /* queue for tasks */
        bool closed{false};                    /* no new tasks should be added */

        bool empty() const { return queue.empty(); }
    };
    mbgl::unordered_map<util::SimpleIdentity, std::shared_ptr<Queue>> taggedQueue;
};

/**
 * @brief ThreadScheduler implements Scheduler interface using a lightweight event loop
 *
 * @tparam N number of threads
 *
 * Note: If N == 1 all scheduled tasks are guaranteed to execute consequently;
 * otherwise, some of the scheduled tasks might be executed in parallel.
 */
class ThreadedScheduler : public ThreadedSchedulerBase {
public:
    ThreadedScheduler(std::size_t n, std::string name = "Worker")
        : ThreadedSchedulerBase(n, std::move(name)),
          threads(n) {
        for (std::size_t i = 0u; i < threads.size(); ++i) {
            threads[i] = makeSchedulerThread(i);
        }
    }

    ~ThreadedScheduler() override {
        assert(!thisThreadIsOwned());
        terminate();
        for (auto& thread : threads) {
            assert(std::this_thread::get_id() != thread.get_id());
            thread.join();
        }
    }

    std::size_t getThreadCount() const noexcept override { return threads.size(); }

    void runOnRenderThread(const util::SimpleIdentity tag, std::function<void()>&& fn) override {
        std::shared_ptr<RenderQueue> queue;
        {
            std::lock_guard lock(taggedRenderQueueLock);
            auto result = taggedRenderQueue.try_emplace(tag);
            if (result.second) {
                // new entry added
                result.first->second = std::make_shared<RenderQueue>();
#ifdef MLN_TRACY_ENABLE
                auto lockName = schedulerName + util::toString(uniqueID) + " renderq" + util::toString(tag);
                MLN_LOCK_NAME_STR(result.first->second->mutex, lockName);
#endif
            }
            queue = result.first->second;
        }

        std::lock_guard lock{queue->mutex};
        queue->queue.push(std::move(fn));
    }

    void runRenderJobs(const util::SimpleIdentity tag, bool closeQueue = false) override {
        MLN_TRACE_FUNC();
        std::shared_ptr<RenderQueue> queue;
        std::unique_lock lock{taggedRenderQueueLock};

        {
            auto it = taggedRenderQueue.find(tag);
            if (it != taggedRenderQueue.end()) {
                queue = it->second;
            }

            if (!closeQueue) {
                lock.unlock();
            }
        }

        if (!queue) {
            return;
        }

        std::lock_guard taskLock{queue->mutex};
        while (queue->queue.size()) {
            auto fn = std::move(queue->queue.front());
            queue->queue.pop();
            if (fn) {
                MLN_TRACE_ZONE(render job);
                fn();
            }
        }

        if (closeQueue) {
            // We hold both locks and can safely remove the queue entry
            taggedRenderQueue.erase(tag);
        }
    }

    mapbox::base::WeakPtr<Scheduler> makeWeakPtr() override { return weakFactory.makeWeakPtr(); }

private:
    std::vector<std::thread> threads;

    struct RenderQueue {
        std::queue<std::function<void()>> queue;
        MLN_TRACE_LOCKABLE(std::mutex, mutex);
    };
    mbgl::unordered_map<util::SimpleIdentity, std::shared_ptr<RenderQueue>> taggedRenderQueue;
    MLN_TRACE_LOCKABLE(std::mutex, taggedRenderQueueLock);

    mapbox::base::WeakPtrFactory<Scheduler> weakFactory{this};
    // Do not add members here, see `WeakPtrFactory`
};

class SequencedScheduler : public ThreadedScheduler {
public:
    SequencedScheduler(std::string name)
        : ThreadedScheduler(1, std::move(name)) {}
};

class ParallelScheduler : public ThreadedScheduler {
public:
    ParallelScheduler(std::size_t extra, std::string name)
        : ThreadedScheduler(1 + extra, std::move(name)) {}
};

class ThreadPool : public ParallelScheduler {
public:
    ThreadPool(std::string name)
        : ParallelScheduler(3, std::move(name)) {}
};

} // namespace mbgl
