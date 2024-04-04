#include "tasksys.h"
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
  // You do not need to implement this method.
  return 0;
}

void TaskSystemSerial::sync() {
  // You do not need to implement this method.
  return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char *TaskSystemParallelSpawn::name() {
  return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads)
    : ITaskSystem(num_threads), num_threads_(num_threads),
      threads_(new std::thread[num_threads]) {}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() { delete[] threads_; }

void TaskSystemParallelSpawn::run(IRunnable *runnable, int num_total_tasks) {
  std::atomic_int task_id(0);
  for (int i = 0; i < num_threads_; ++i) {
    threads_[i] = std::thread([&] {
      while (true) {
        int id = task_id.fetch_add(1);
        if (id >= num_total_tasks) {
          break;
        }
        runnable->runTask(id, num_total_tasks);
      }
    });
  }

  for (int i = 0; i < num_threads_; ++i) {
    threads_[i].join();
  }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {
  // You do not need to implement this method.
  return 0;
}

void TaskSystemParallelSpawn::sync() {
  // You do not need to implement this method.
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
    : ITaskSystem(num_threads), num_threads_(num_threads),
      threads_(new std::thread[num_threads]), end_(false), runnable_(nullptr) {
  for (int i = 0; i < num_threads; ++i) {
    threads_[i] = std::thread([&] {
      while (true) {
        IRunnable *runnable = nullptr;
        int task_id;
        mutex_.lock();
        if (runnable_) {
          runnable = runnable_;
          task_id = current_task_id_++;
          if (current_task_id_ >= num_total_tasks_) {
            runnable_ = nullptr;
          }
        }
        mutex_.unlock();
        if (runnable) {
          runnable->runTask(task_id, num_total_tasks_);
          num_finished_task_.fetch_add(1);
        } else if (end_) {
          break;
        }
      }
    });
  }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
  end_ = true;
  for (int i = 0; i < num_threads_; ++i) {
    threads_[i].join();
  }
  delete[] threads_;
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable *runnable,
                                               int num_total_tasks) {
  mutex_.lock();
  runnable_ = runnable;
  num_total_tasks_ = num_total_tasks;
  current_task_id_ = 0;
  num_finished_task_ = 0;
  mutex_.unlock();
  while (num_finished_task_ < num_total_tasks) {
    // spin
  }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {
  // You do not need to implement this method.
  return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
  // You do not need to implement this method.
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
    : ITaskSystem(num_threads), num_threads_(num_threads),
      threads_(new std::thread[num_threads]), end_(false), runnable_(nullptr),
      num_sleeping_threads_(0) {
  for (int i = 0; i < num_threads; ++i) {
    threads_[i] = std::thread([&] {
      while (true) {
        if (end_) {
          break;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        if (IRunnable *runnable = runnable_) {
          int task_id = current_task_id_++;
          if (current_task_id_ >= num_total_tasks_) {
            runnable_ = nullptr;
          }
          lock.unlock();
          runnable->runTask(task_id, num_total_tasks_);
        } else {
          num_sleeping_threads_++;
          cv_.wait(lock);
          num_sleeping_threads_--;
        }
      }
    });
  }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
  end_ = true;
  while (num_sleeping_threads_ > 0) {
    cv_.notify_all();
  }
  for (int i = 0; i < num_threads_; ++i) {
    threads_[i].join();
  }
  delete[] threads_;
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable *runnable,
                                               int num_total_tasks) {
  std::unique_lock<std::mutex> lock(mutex_);
  runnable_ = runnable;
  num_total_tasks_ = num_total_tasks;
  current_task_id_ = 0;
  lock.unlock();
  cv_.notify_all();
  while (runnable_ || num_sleeping_threads_ < num_threads_) {
    // spin
  }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {

  //
  // TODO: CS149 students will implement this method in Part B.
  //

  return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

  //
  // TODO: CS149 students will modify the implementation of this method in Part
  // B.
  //

  return;
}
