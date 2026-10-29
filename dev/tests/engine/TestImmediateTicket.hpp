#pragma once

#include "engine/Types.hpp"

#include <atomic>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::test {

// The times a command reports (ModelBatchTicket): none unless a test sets
// them, as a fake runs no kernels.
struct CommandTimes final {
  double wallMilliseconds = 0.0;
  double hostMilliseconds = 0.0;
  double streamMilliseconds = 0.0;
};

class ImmediateTicket final : public ModelBatchTicket {
public:
  explicit ImmediateTicket(std::vector<ModelStepResult> results, CommandTimes times = {})
      : results_(std::move(results)), times_(times) {}

  [[nodiscard]] bool ready() const noexcept override { return true; }
  [[nodiscard]] std::vector<ModelStepResult> wait() override {
    return std::move(results_);
  }
  [[nodiscard]] double wallMilliseconds() const noexcept override {
    return times_.wallMilliseconds;
  }
  [[nodiscard]] double hostMilliseconds() const noexcept override {
    return times_.hostMilliseconds;
  }
  [[nodiscard]] double streamMilliseconds() const noexcept override {
    return times_.streamMilliseconds;
  }

private:
  std::vector<ModelStepResult> results_;
  CommandTimes times_;
};

// A command that stays in flight until the test sets `ready`, from any
// thread.
class HeldTicket final : public ModelBatchTicket {
public:
  HeldTicket(std::vector<ModelStepResult> results,
             std::shared_ptr<std::atomic<bool>> ready, double wallMilliseconds)
      : results_(std::move(results)), ready_(std::move(ready)),
        wallMilliseconds_(wallMilliseconds) {}

  [[nodiscard]] bool ready() const noexcept override { return *ready_; }
  [[nodiscard]] std::vector<ModelStepResult> wait() override {
    return std::move(results_);
  }
  [[nodiscard]] double wallMilliseconds() const noexcept override {
    return wallMilliseconds_;
  }

private:
  std::vector<ModelStepResult> results_;
  std::shared_ptr<std::atomic<bool>> ready_;
  double wallMilliseconds_ = 0.0;
};

// A command of two, as a prefill command whose images are encoded ahead of
// it (model::Runtime): the first runs until the test sets `first`, from any
// thread, and the second completes once the engine's advance() has run it
// after that.
class StagedTicket final : public ModelBatchTicket {
public:
  StagedTicket(std::vector<ModelStepResult> results, std::shared_ptr<std::atomic<bool>> first)
      : results_(std::move(results)), first_(std::move(first)) {}

  void advance() override { second_ = second_ || *first_; }
  [[nodiscard]] bool ready() const noexcept override { return second_; }
  [[nodiscard]] std::vector<ModelStepResult> wait() override {
    if (!second_)
      throw std::logic_error("a staged ticket completed before its second command ran");
    return std::move(results_);
  }
  [[nodiscard]] double wallMilliseconds() const noexcept override { return 1.0; }

private:
  std::vector<ModelStepResult> results_;
  std::shared_ptr<std::atomic<bool>> first_;
  bool second_ = false;
};

inline std::unique_ptr<ModelBatchTicket>
immediateTicket(std::vector<ModelStepResult> results,
                const std::function<void()> &completion, CommandTimes times = {}) {
  if (completion)
    completion();
  return std::make_unique<ImmediateTicket>(std::move(results), times);
}

} // namespace splash::test
