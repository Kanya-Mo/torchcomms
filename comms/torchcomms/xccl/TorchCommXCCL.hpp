// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <sycl/sycl.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommBackend.hpp"
#include "comms/torchcomms/TorchCommBatch.hpp"
#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/xccl/TorchWorkXCCL.hpp"
#include <oneapi/ccl.h>

namespace torch {
namespace comms {

constexpr size_t kDefaultMaxEventPoolSize = 1000;

class TorchCommXCCL : public TorchCommBackend,
                      public std::enable_shared_from_this<TorchCommXCCL> {
 public:
  static constexpr std::string_view kBackendName = "xccl";

  TorchCommXCCL();
  ~TorchCommXCCL() override;

  // Delete copy and move operations
  TorchCommXCCL(const TorchCommXCCL&) = delete;
  TorchCommXCCL(TorchCommXCCL&&) = delete;
  TorchCommXCCL& operator=(const TorchCommXCCL&) = delete;
  TorchCommXCCL& operator=(TorchCommXCCL&&) = delete;

//   void init(
//       at::Device device,
//       const std::string& name,
//       const CommOptions& options = {}) override;
  void finalize() override;
  int getRank() const override;
  int getSize() const override;
  std::string_view getBackendName() const override;
  std::string_view getCommName() const override;

  // Point-to-Point Operations
  c10::intrusive_ptr<TorchWork> send(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      const SendOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> recv(
      at::Tensor& tensor,
      int src,
      bool async_op,
      const RecvOptions& options = {}) override;

  // Batch P2P Operations
  c10::intrusive_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      const BatchP2POptions& options = {}) override;

  // Collective Operations
  c10::intrusive_ptr<TorchWork> broadcast(
      at::Tensor& tensor,
      int root,
      bool async_op,
      const BroadcastOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      const ReduceOp& op,
      bool async_op,
      const AllReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce(
      const at::Tensor& tensor,
      int root,
      const ReduceOp& op,
      bool async_op,
      const ReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_v(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllGatherSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor& output,
      const at::Tensor& input,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllToAllSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      const AllToAllvSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      const AllToAllOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> barrier(
      bool async_op,
      const BarrierOptions& options = {}) override;

  // Scatter and Gather Operations
  c10::intrusive_ptr<TorchWork> scatter(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      const ScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> gather(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      const GatherOptions& options = {}) override;

  // Friend access for TorchCommXCCL
  friend class TorchWorkXCCL;
  
  const CommOptions& getOptions() const override {
    return options_;
  }

  const at::Device& getDevice() const override {
    return device_;
  }

 protected:
  c10::intrusive_ptr<TorchWorkXCCL> createWork(
      // cudaStream_t stream,
      std::chrono::milliseconds timeout,
      const std::vector<at::Tensor>& inputTensors);

 private:
  // Constructor for split communicators
  explicit TorchCommXCCL(const onecclComm_t xccl_comm);
  onecclComm_t xccl_comm_{};
  at::Device device_;
  CommOptions options_;
  std::atomic<bool> shutdown_;
};

}
}