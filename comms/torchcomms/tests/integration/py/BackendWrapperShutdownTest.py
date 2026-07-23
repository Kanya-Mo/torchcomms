#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Verify ``BackendWrapper::shutdown`` and ``::abort`` close the underlying
``TorchComm`` cleanly when ``torch.distributed.destroy_process_group()``
is called.

Two regressions guarded against:

1. **Deadlock on destroy**: without a ``shutdown`` override the wrapper's
   destructor would invoke ``ncclCommDestroy`` synchronously and could
   deadlock against the NCCL GC thread.

2. **Double-finalize raise**: a mixed ``cpu:gloo,cuda:nccl`` PG ends up
   with two BackendWrappers sharing one underlying ``TorchComm`` (via the
   BackendType-to-wrapper dedup). ``destroy_process_group`` calls
   ``shutdown`` on each backend; ``TorchComm::finalize`` is not idempotent
   and would raise ``RuntimeError: TorchCommNCCL already finalized`` on
   the second call. The wrapper swallows the exception so destroy is safe
   to call any number of times.
"""

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
from packaging.version import InvalidVersion, Version
from torchcomms.tests.helpers.py.test_helpers import skip_if_ncclx
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    get_device,
    get_rank_and_size,
)

_TORCHCOMMS_CONFIG_AVAILABLE = hasattr(dist, "config") and hasattr(
    dist.config, "use_torchcomms"
)

_PR_182057_TORCH_VERSION = Version("2.13.0.dev20260502")


def _torch_predates_pr_182057() -> bool:
    try:
        return Version(torch.__version__) < _PR_182057_TORCH_VERSION
    except InvalidVersion:
        # Unparsable version string — assume newer to avoid silently
        # skipping on something we can't classify.
        return False


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", get_rank_and_size()[0]))


# Each test gets its own rendezvous port so that reusing the launcher's
# single ``MASTER_PORT`` across multiple ``init_process_group`` calls in one
# process does not race (the original flakiness this test guarded against).
#
# The port is a deterministic function of ``MASTER_PORT`` plus a fixed
# per-test offset, so every rank computes the same value independently with
# no cross-rank hand-off. This is what makes it multi-node safe: a node-local
# file (the previous approach) is invisible to ranks on other hosts, whereas
# a pure function of the shared launcher env agrees everywhere.
#
# Offsets are an explicit table rather than a hash of the name: the set of
# store names is small and fixed, so this is collision-free and auditable,
# and it avoids ``hash()`` (randomized per process via ``PYTHONHASHSEED``,
# which would make ranks disagree).
_PORT_OFFSETS = {
    "destroy_after_collective_no_hang": 1,
    "mixed_backend_destroy_idempotent": 2,
}


def _store_port(store_name: str) -> int:
    base = int(os.environ["MASTER_PORT"])
    # ``+ offset`` keeps every test clear of ``MASTER_PORT`` itself (offsets
    # start at 1). Wrap back into the unprivileged range if we would exceed
    # the 16-bit port space.
    port = base + _PORT_OFFSETS[store_name]
    if port > 65535:
        port = 1024 + (port % (65535 - 1024))
    return port


def _create_isolated_store(store_name: str) -> dist.TCPStore:
    rank, world_size = get_rank_and_size()
    return dist.TCPStore(
        host_name=os.environ["MASTER_ADDR"],
        port=_store_port(store_name),
        world_size=world_size,
        is_master=(rank == 0),
        wait_for_workers=False,
    )


@unittest.skipUnless(
    _TORCHCOMMS_CONFIG_AVAILABLE,
    "dist.config.use_torchcomms not available in this PyTorch version",
)
@skip_if_ncclx
class TestBackendWrapperShutdown(unittest.TestCase):
    """Each test creates its own PG, runs a small collective, then tears
    it down — no shared setUpClass, since the goal is to exercise the
    init+destroy cycle itself."""

    def _init_pg(self, backend: str, store_name: str) -> None:
        rank, world_size = get_rank_and_size()
        # NCCL requires a CUDA device to be bound to this rank before
        # ``init_process_group`` so that the per-rank communicator gets
        # the right device. Without this, all ranks default to cuda:0
        # and NCCL bootstrap fails. ``get_device`` can return
        # ``torch.device("cuda")`` with no index when ``TEST_DEVICE=cuda``
        # is set, so resolve the index explicitly from LOCAL_RANK / rank.
        device = get_device(os.environ["TEST_BACKEND"], rank)
        if torch.accelerator.is_available():
            torch.accelerator.set_device_index(_local_rank())
        dist.config.use_torchcomms = True
        dist.init_process_group(
            backend=backend,
            store=_create_isolated_store(store_name),
            rank=rank,
            world_size=world_size,
        )
        torch.set_default_device(device)

    def test_destroy_after_collective_no_hang(self):
        """A simple init → all_reduce → destroy cycle finishes without
        hanging. Catches the original ``ncclCommDestroy`` deadlock."""
        store_name = "destroy_after_collective_no_hang"
        self._init_pg(os.environ["TEST_BACKEND"], store_name)
        try:
            tensor = torch.ones(8, dtype=torch.float32)
            dist.all_reduce(tensor)
            self.assertEqual(tensor[0].item(), float(dist.get_world_size()))
        finally:
            dist.destroy_process_group()

    def test_mixed_backend_destroy_idempotent(self):
        """Mixed ``cpu:gloo,cuda:nccl`` PG: ``destroy_process_group``
        shuts down both sub-backends, which share one ``TorchComm``.
        Without idempotent ``shutdown``, the second call raises
        ``TorchCommNCCL already finalized``."""
        if os.environ["TEST_BACKEND"] != "nccl":
            self.skipTest("mixed backend test is nccl-specific")
        if _torch_predates_pr_182057():
            self.skipTest(
                f"torch {torch.__version__} predates pytorch/pytorch#182057 "
                f"({_PR_182057_TORCH_VERSION}); mixed cpu:gloo,cuda:nccl + "
                "device_id= trips the ProcessGroup::setBackend "
                "bound_device_id check on init"
            )

        rank, world_size = get_rank_and_size()
        local_rank = _local_rank()
        torch.cuda.set_device(local_rank)
        dist.config.use_torchcomms = True
        store_name = "mixed_backend_destroy_idempotent"
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            store=_create_isolated_store(store_name),
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        try:
            torch.set_default_device(f"cuda:{local_rank}")
            cpu_tensor = torch.ones(4, dtype=torch.float32, device="cpu")
            cuda_tensor = torch.ones(
                4, dtype=torch.float32, device=f"cuda:{local_rank}"
            )
            dist.all_reduce(cpu_tensor)
            dist.all_reduce(cuda_tensor)
            self.assertEqual(cpu_tensor[0].item(), float(world_size))
            self.assertEqual(cuda_tensor[0].item(), float(world_size))
        finally:
            # Must not raise even though both sub-backends share the comm.
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
