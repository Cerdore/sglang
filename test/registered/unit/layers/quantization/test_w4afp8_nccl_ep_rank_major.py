"""CPU guard: NCCL EP rank-major output feeds the masked W4AFP8 GEMM.

The rank-major dispatch view is a 9-field NamedTuple, so passing it to the
DeepEP low-latency adapter (which unpacks exactly 6 fields) fails before any
kernel runs. The adapter must select its inputs by field name instead.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
    NcclEpRankMajorDispatchOutput,
)
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.test.test_utils import CustomTestCase


def _fake_layer():
    quant_method = SimpleNamespace(
        a_strides1=1,
        b_strides1=2,
        c_strides1=3,
        a_strides2=4,
        b_strides2=5,
        c_strides2=6,
        s_strides13=7,
        s_strides2=8,
        expert_offsets=9,
        problem_sizes1=10,
        problem_sizes2=11,
    )
    return SimpleNamespace(
        w13_weight="w13",
        w2_weight="w2",
        w13_weight_scale_inv="w13_scale",
        w2_weight_scale_inv="w2_scale",
        w13_input_scale="w13_input",
        w2_input_scale="w2_input",
        quant_method=quant_method,
    )


class TestNcclEpRankMajorW4AFp8(CustomTestCase):
    def test_adapter_forwards_packed_view_to_masked_gemm(self):
        method = W4AFp8MoEMethod(W4AFp8Config())
        layer = _fake_layer()

        hidden = torch.zeros(8, 16, dtype=torch.float8_e4m3fn)
        hidden_scale = torch.ones(8, 2, dtype=torch.float32)
        topk_ids = torch.zeros(8, 2, dtype=torch.int32)
        masked_m = torch.tensor([4, 4], dtype=torch.int32)
        dispatch_output = NcclEpRankMajorDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=hidden_scale,
            topk_ids=topk_ids,
            masked_m=masked_m,
            expected_m=3,
            src2dst=torch.zeros(4, dtype=torch.int32),
            local_topk_ids=topk_ids,
            recv_topk_weights=torch.ones(8, 2, dtype=torch.float32),
            slot_shape=(2, 4),
        )

        with patch(
            "sglang.srt.layers.moe.cutlass_w4a8_moe.cutlass_w4a8_moe_deepep_ll",
            return_value="gemm_out",
        ) as gemm:
            out = method.apply_nccl_ep_rank_major(layer, dispatch_output)

        self.assertEqual(out, "gemm_out")
        args = gemm.call_args.args
        self.assertIs(args[0], hidden)
        self.assertIs(args[1], hidden_scale)
        self.assertIs(args[6], topk_ids)
        self.assertIs(args[7], masked_m)
        self.assertEqual(gemm.call_args.kwargs["expected_m"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=3)
