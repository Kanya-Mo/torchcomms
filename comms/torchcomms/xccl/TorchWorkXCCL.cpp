// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/xccl/TorchWorkXCCL.hpp"
#include <ATen/xpu/XPUContext.h>
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/xccl/TorchCommXCCL.hpp"

namespace torch {
namespace comms {

TorchWorkXCCL::TorchWorkXCCL(
    std::shared_ptr<TorchCommNCCL> comm,
    // cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const std::vector<at::Tensor>& inputTensors,
    std::shared_ptr<TorchCommTracing> tracing)
    : inputTensors_(inputTensors),
      comm_(std::move(comm)),
      // stream_(stream),
      timeout_ms_(timeout_ms),
      state_(WorkStatus::NOT_STARTED),
      tracing_(std::move(tracing)) {
}

TorchWorkXCCL::TorchWorkXCCL(
    std::shared_ptr<TorchCommNCCL> comm,
    // cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const at::Tensor& inputTensor,
    std::shared_ptr<TorchCommTracing> tracing)
    : inputTensor_(inputTensor),
      comm_(std::move(comm)),
      // stream_(stream),
      timeout_ms_(timeout_ms),
      state_(WorkStatus::NOT_STARTED),
      tracing_(std::move(tracing)) {
}

TorchWorkXCCL::~TorchWorkXCCL() {}

bool TorchWorkXCCL::isCompleted() {
  return state_ == WorkStatus::COMPLETED;
}

TorchWorkXCCL::WorkStatus TorchWorkXCCL::checkStatus() {
  return state_;
}

void TorchWorkXCCL::wait() {
  // If already completed, return immediately
  state_ = WorkStatus::COMPLETED; // fake
  return;
}
} // namespace comms
} // namespace torch
