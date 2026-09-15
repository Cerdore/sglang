"""Direct NVLS q-gather parity for the symmetric-memory DCP backend.

This is the CI anchor for the Phase-2 direct q-gather (sglang port of vLLM
#50484). The q-gather is NOT a separate ``--dcp-comm-backend``; it is
auto-enabled inside ``symm_a2a`` whenever the runtime NVLS multicast probe
(``symm_mem.multicast_ptr != 0``) succeeds, replacing the per-layer NCCL Q
AllGather. On topologies without NVSwitch (e.g. A100, or 2-GPU direct NVLink
without a switch) it falls back to NCCL automatically and this test still
asserts parity -- it just does not exercise the multicast kernel on those
boxes.

Baseline is the ``a2a`` (NCCL All-to-All) backend rather than ``ag_rs`` so the
comparison isolates "direct symmetric-memory + multimem q-gather" against
"NCCL A2A + NCCL Q AllGather" -- the two direct-A2A backends. See
``docs/phase2_qgather_test_plan.md`` for the full verification matrix, including
the manual stderr probe that asserts one of the q-gather enable/fallback log
lines fires (proving the new dispatch path was reached).
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci

from test_dcp_symm_a2a import SymmA2ATestBase

register_cuda_ci(est_time=540, stage="base-b", runner_config="2-gpu-large")


class TestDCPQGatherTP2(SymmA2ATestBase):
    required_gpus = 2

    def test_symm_a2a_with_q_gather_matches_a2a(self):
        """symm_a2a (q-gather auto-enabled on NVSwitch) vs a2a (NCCL)."""
        for disable_cuda_graph in (False, True):
            with self.subTest(disable_cuda_graph=disable_cuda_graph):
                baseline = self._run_case(
                    tp_size=2,
                    backend="a2a",
                    disable_cuda_graph=disable_cuda_graph,
                )
                actual = self._run_case(
                    tp_size=2,
                    backend="symm_a2a",
                    disable_cuda_graph=disable_cuda_graph,
                )
                self.assertIsNotNone(baseline.outputs)
                self.assertIsNotNone(actual.outputs)
                self._assert_backend_parity(baseline.outputs, actual.outputs)


if __name__ == "__main__":
    unittest.main()
