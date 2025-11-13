// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>

#include <ATen/ATen.h>
#include <sycl/sycl.hpp>
#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/TorchWork.hpp"

namespace torch {
namespace comms {

// Forward declaration
class TorchCommXCCL;

class TorchWorkXCCL : public TorchWork {
 public:
  // Status of a work object
  enum class WorkStatus {
    NOT_STARTED, // Work has not started yet
    INPROGRESS, // Work is still in progress,
    COMPLETED, // Work has completed successfully
    TIMEDOUT, // Work has timed out
    ERROR // Work has encountered an error
  };

  TorchWorkXCCL(
      std::shared_ptr<TorchCommNCCL> comm,
      // sycl::queue& stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors,
      std::shared_ptr<TorchCommTracing> tracing);
  TorchWorkXCCL(
      std::shared_ptr<TorchCommNCCL> comm,
      // sycl::queue& stream,
      std::chrono::milliseconds timeout_ms,
      const at::Tensor& inputTensor,
      std::shared_ptr<TorchCommTracing> tracing);
  ~TorchWorkXCCL() override;

  // Delete copy and move operations
  TorchWorkXCCL(const TorchWorkXCCL&) = delete;
  TorchWorkXCCL(TorchWorkXCCL&&) = delete;
  TorchWorkXCCL& operator=(const TorchWorkXCCL&) = delete;
  TorchWorkXCCL& operator=(TorchWorkXCCL&&) = delete;

  // Override virtual functions from TorchWork
  bool isCompleted() override;
  void wait() override;

  friend class TorchCommXCCL;

 private:

  std::chrono::milliseconds getTimeout() const {
    return timeout_ms_;
  }
  std::vector<at::Tensor> inputTensors_;
  at::Tensor inputTensor_;

  std::shared_ptr<TorchCommNCCL> comm_;
  // sycl::queue& stream_; // stream is not owned by this class

  std::chrono::milliseconds timeout_ms_;

  // state machine variables. TODO: convert to state machine later
  std::atomic<WorkStatus> state_;

  std::shared_ptr<TorchCommTracing> tracing_;
};

} // namespace comms
} // namespace torch
