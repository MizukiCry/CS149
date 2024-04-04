#include "tasksys.h"
#include <algorithm>
#include <iostream>

IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char *TaskSystemSerial::name() { return "Serial"; }

TaskSystemSerial::TaskSystemSerial(int num_threads)
    : ITaskSystem(num_threads) {}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable *runnable, int num_total_tasks) {
  for (int i = 0; i < num_total_tasks; i++) {
    runnable->runTask(i, num_total_tasks);
  }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable *runnable,
                                          int num_total_tasks,
                                          const std::vector<TaskID> &deps) {
  for (int i = 0; i < num_total_tasks; i++) {
    runnable->runTask(i, num_total_tasks);
  }

  return 0;
}

void TaskSystemSerial::sync() { return; }

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char *TaskSystemParallelSpawn::name() {
  return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads)
    : ITaskSystem(num_threads) {
  // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn
  // in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable *runnable, int num_total_tasks) {
  // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn
  // in Part B.
  for (int i = 0; i < num_total_tasks; i++) {
    runnable->runTask(i, num_total_tasks);
  }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {
  // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn
  // in Part B.
  for (int i = 0; i < num_total_tasks; i++) {
    runnable->runTask(i, num_total_tasks);
  }

  return 0;
}

void TaskSystemParallelSpawn::sync() {
  // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn
  // in Part B.
  return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char *TaskSystemParallelThreadPoolSpinning::name() {
  return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(
    int num_threads)
    : ITaskSystem(num_threads) {
  // NOTE: CS149 students are not expected to implement
  // TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable *runnable,
                                               int num_total_tasks) {
  // NOTE: CS149 students are not expected to implement
  // TaskSystemParallelThreadPoolSpinning in Part B.
  for (int i = 0; i < num_total_tasks; i++) {
    runnable->runTask(i, num_total_tasks);
  }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {
  // NOTE: CS149 students are not expected to implement
  // TaskSystemParallelThreadPoolSpinning in Part B.
  for (int i = 0; i < num_total_tasks; i++) {
    runnable->runTask(i, num_total_tasks);
  }

  return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
  // NOTE: CS149 students are not expected to implement
  // TaskSystemParallelThreadPoolSpinning in Part B.
  return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char *TaskSystemParallelThreadPoolSleeping::name() {
  return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(
    int num_threads)
    : ITaskSystem(num_threads), next_task_id_(0), end_(false),
      num_threads_(num_threads), threads_(new std::thread[num_threads]) {
  for (int i = 0; i < num_threads_; ++i) {
    threads_[i] = std::thread([&] {
      while (true) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (ready_tasks_.empty()) {
          if (end_) {
            return;
          }
          cv_.wait(lock);
        }

        auto it = std::find_if(
            ready_tasks_.begin(), ready_tasks_.end(), [](const Task &task) {
              return task.current_task < task.num_total_tasks;
            });
        if (it != ready_tasks_.end()) {
          int task_id = it->current_task++;
          lock.unlock();

          it->runnable->runTask(task_id, it->num_total_tasks);

          lock.lock();
          if (++it->num_finished_tasks == it->num_total_tasks) {
            for (TaskID successor : it->successors) {
              if (--tasks_[successor].num_deps == 0) {
                ready_tasks_.push_back(tasks_[successor]);
                tasks_.erase(successor);
              }
            }
            ready_tasks_.erase(it);
            lock.unlock();
            cv_.notify_all();
          }
        }
      }
    });
  }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
  std::unique_lock<std::mutex> lock(mutex_);
  end_ = true;
  while (!ready_tasks_.empty()) {
    cv_.wait(lock);
  }
  lock.unlock();
  cv_.notify_all();
  for (int i = 0; i < num_threads_; ++i) {
    threads_[i].join();
  }
  delete[] threads_;
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable *runnable,
                                               int num_total_tasks) {
  sync();
  runAsyncWithDeps(runnable, num_total_tasks, {});
  sync();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {
  std::unique_lock<std::mutex> lock(mutex_);
  int task_id = next_task_id_++;
  Task task{runnable, num_total_tasks, 0, 0, {}, 0};
  for (TaskID dep : deps) {
    auto it = tasks_.find(dep);
    if (it != tasks_.end()) {
      it->second.successors.push_back(task_id);
      task.num_deps++;
    }
  }
  if (task.num_deps == 0) {
    ready_tasks_.push_back(task);
  } else {
    tasks_.emplace(task_id, task);
  }
  lock.unlock();
  cv_.notify_all();
  return task_id;
}

void TaskSystemParallelThreadPoolSleeping::sync() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!ready_tasks_.empty()) {
    cv_.wait(lock);
  }
}
