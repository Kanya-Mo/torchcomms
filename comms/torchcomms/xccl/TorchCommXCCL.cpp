// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/xccl/TorchCommXCCL.hpp"

#include <ATen/xpu/XPUContext.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"

namespace torch {
namespace comms {

TorchCommXCCL::TorchCommXCCL()
    : xccl_comm_{nullptr},
      device_(at::kXPU),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

TorchCommXCCL::TorchCommXCCL(const onecclComm_t xccl_comm)
    : xccl_comm_(xccl_comm),
      device_(at::kCUDA),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

TorchCommXCCL::~TorchCommXCCL() {
  if (init_state_ == InitializationState::INITIALIZED) {
    TC_LOG(ERROR) << "TorchCommXCCL was not finalized before destruction";
  }
}

void TorchCommXCCL::finalize() {
  shutdown_ = true;
  // Destroy oneCCL communicator.
  if (xccl_comm_) {
    // oneccl_api_->commDestroy(xccl_comm_);
    xccl_comm_ = nullptr;
  }
}

int TorchCommXCCL::getRank() const {
  return 0;
}

int TorchCommXCCL::getSize() const {
  return 1;
}

std::string_view TorchCommXCCL::getBackendName() const {
  return kBackendName;
}

std::string_view TorchCommXCCL::getCommName() const {
  return "fake_xccl";
}

c10::intrusive_ptr<TorchWorkXCCL> TorchCommXCCL::createWork(
    // cudaStream_t stream,
    std::chrono::milliseconds timeout,
    const std::vector<at::Tensor>& inputTensors) {
  // Only create the work object without enqueuing it
  auto work = c10::make_intrusive<TorchWorkXCCL>(
      shared_from_this(), timeout, inputTensors, tracing_); // stream,
  return work;
}

// Point-to-Point Operations
c10::intrusive_ptr<TorchWork> TorchCommXCCL::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    const BatchP2POptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

// Collective Operations
c10::intrusive_ptr<TorchWork> TorchCommXCCL::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  throw std::runtime_error("all_gather_v is not supported in XCCL backend"); // not sure 
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter_v(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  throw std::runtime_error("reduce_scatter_v is not supported in XCCL backend"); // not sure 
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::barrier(
    bool async_op,
    const BarrierOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

std::shared_ptr<TorchCommBackend> TorchCommXCCL::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  auto work = createWork(
      getOperationTimeout(options.timeout, options_.timeout), {tensor});
  return work;
}

}
}

namespace {
class XCCLRegistration {
 public:
  XCCLRegistration() {
    torch::comms::TorchCommFactory::get().register_backend("xccl", []() {
      return std::make_shared<torch::comms::TorchCommXCCL>();
    });
  }
};

static XCCLRegistration registration{};
} // namespace