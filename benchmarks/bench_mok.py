import os

import torch.distributed as dist

from benchmarks.utils import benchmark_bwd, benchmark_fwd, check_benchmark_correctness, get_num_local_experts, get_tflops, init_distributed
from mok import functional, ops
from tests.utils import BF16_TOLERANCE, MXFP8_TOLERANCE, generate_inputs, run_reference_bf16


NUM_LOCAL_TOKENS = int(os.environ.get("NUM_LOCAL_TOKENS", 2048))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
INTERMEDIATE_DIM = int(os.environ.get("INTERMEDIATE_DIM", 3072))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 384))
TOPK = int(os.environ.get("TOPK", 6))
BF16_FWD_COMM_SMS = int(os.environ.get("BF16_FWD_COMM_SMS", 24))
BF16_BWD_COMM_SMS = int(os.environ.get("BF16_BWD_COMM_SMS", 28))
MXFP8_FWD_COMM_SMS = int(os.environ.get("MXFP8_FWD_COMM_SMS", 36))
MXFP8_BWD_COMM_SMS = int(os.environ.get("MXFP8_BWD_COMM_SMS", 36))
MINIBATCH_SIZE = int(os.environ.get("MINIBATCH_SIZE", 4096))
MACROBATCH_SIZE = int(os.environ.get("MACROBATCH_SIZE", 32 * MINIBATCH_SIZE))
# Comma-separated subset of BF16, MXFP8, BLOCKWISE (blockwise FP8: 1x128 activation / 128x128 weight scales)
VARIANTS = tuple(v.strip().upper() for v in os.environ.get("VARIANTS", "BF16,MXFP8,BLOCKWISE").split(",") if v.strip())
BLOCKWISE_TOLERANCE = MXFP8_TOLERANCE

BF16_CONFIG = functional.MoKConfig(
    fwd_num_comm_sms=BF16_FWD_COMM_SMS,
    bwd_num_comm_sms=BF16_BWD_COMM_SMS,
    minibatch_size=MINIBATCH_SIZE,
    macrobatch_size=MACROBATCH_SIZE,
)
MXFP8_CONFIG = functional.MoKConfig(
    fwd_num_comm_sms=MXFP8_FWD_COMM_SMS,
    bwd_num_comm_sms=MXFP8_BWD_COMM_SMS,
    minibatch_size=MINIBATCH_SIZE,
    macrobatch_size=MACROBATCH_SIZE,
)
# Same tuning as MXFP8: the two recipes share the FP8 GEMM mainloop and differ only in scale granularity
BLOCKWISE_CONFIG = functional.MoKConfig(
    fwd_num_comm_sms=MXFP8_FWD_COMM_SMS,
    bwd_num_comm_sms=MXFP8_BWD_COMM_SMS,
    minibatch_size=MINIBATCH_SIZE,
    macrobatch_size=MACROBATCH_SIZE,
    fp8_recipe="blockwise",
)


class MoKBenchmark:
    def __init__(self, inputs):
        (
            self.x,
            self.topk_experts,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            self.w_routed_gate,
            self.w_routed_up,
            self.w_routed_down,
            self.d_output,
        ) = inputs
        self.num_local_experts = self.w_routed_gate.shape[0]
        self.workspace = functional.get_workspace(BF16_CONFIG, dist.group.WORLD, device=self.x.device, num_local_tokens=self.x.shape[0], hidden_size=self.x.shape[1], topk=self.topk_experts.shape[1])
        if "MXFP8" in VARIANTS:
            self.w_gate_mxfp8 = ops.mxfp8_quantize(self.w_routed_gate, True, True)
            self.w_up_mxfp8 = ops.mxfp8_quantize(self.w_routed_up, True, True)
            self.w_down_mxfp8 = ops.mxfp8_quantize(self.w_routed_down, True, True)
        if "BLOCKWISE" in VARIANTS:
            self.w_gate_blockwise = ops.blockwise_fp8_quantize(self.w_routed_gate, True, True, 128)
            self.w_up_blockwise = ops.blockwise_fp8_quantize(self.w_routed_up, True, True, 128)
            self.w_down_blockwise = ops.blockwise_fp8_quantize(self.w_routed_down, True, True, 128)

    def run_bf16_fwd(self):
        schedule = functional.build_schedule(self.workspace, BF16_CONFIG, self.topk_experts, num_local_experts=self.num_local_experts)
        output, forward_context = functional.forward(
            BF16_CONFIG,
            self.workspace,
            schedule,
            self.x,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            self.w_routed_gate,
            self.w_routed_up,
            self.w_routed_down,
        )
        return output, (schedule, forward_context)

    def run_bf16_bwd(self, context):
        schedule, forward_context = context
        return functional.backward(
            BF16_CONFIG,
            self.workspace,
            schedule,
            forward_context,
            self.d_output,
            self.x,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            self.w_routed_gate,
            self.w_routed_up,
            self.w_routed_down,
        )

    def _run_fp8_fwd(self, config, w_gate, w_up, w_down):
        schedule = functional.build_schedule(self.workspace, config, self.topk_experts, num_local_experts=self.num_local_experts)
        output, forward_context = functional.forward(
            config,
            self.workspace,
            schedule,
            self.x,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            w_gate[:2],
            w_up[:2],
            w_down[:2],
        )
        return output, (schedule, forward_context)

    def _run_fp8_bwd(self, config, w_gate, w_up, w_down, context):
        schedule, forward_context = context
        return functional.backward(
            config,
            self.workspace,
            schedule,
            forward_context,
            self.d_output,
            self.x,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            w_gate,
            w_up,
            w_down[2:],
        )

    def run_mxfp8_fwd(self):
        return self._run_fp8_fwd(MXFP8_CONFIG, self.w_gate_mxfp8, self.w_up_mxfp8, self.w_down_mxfp8)

    def run_mxfp8_bwd(self, context):
        return self._run_fp8_bwd(MXFP8_CONFIG, self.w_gate_mxfp8, self.w_up_mxfp8, self.w_down_mxfp8, context)

    def run_blockwise_fwd(self):
        return self._run_fp8_fwd(BLOCKWISE_CONFIG, self.w_gate_blockwise, self.w_up_blockwise, self.w_down_blockwise)

    def run_blockwise_bwd(self, context):
        return self._run_fp8_bwd(BLOCKWISE_CONFIG, self.w_gate_blockwise, self.w_up_blockwise, self.w_down_blockwise, context)


def main() -> None:
    rank, world_size, device = init_distributed()
    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(rank, device, NUM_EXPERTS, num_local_experts, TOPK, NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM)
    benchmark = MoKBenchmark(inputs)

    if rank == 0:
        print(f"tokens/rank={NUM_LOCAL_TOKENS} experts={NUM_EXPERTS} ({num_local_experts}/rank) topk={TOPK} H={HIDDEN_DIM} I={INTERMEDIATE_DIM} EP={world_size}")
        print(f"BF16 comm SMs={BF16_FWD_COMM_SMS}/{BF16_BWD_COMM_SMS}, MXFP8/BLOCKWISE comm SMs={MXFP8_FWD_COMM_SMS}/{MXFP8_BWD_COMM_SMS}, minibatch={MINIBATCH_SIZE}, macrobatch={MACROBATCH_SIZE}")
        print(f"variants={','.join(VARIANTS)}")

    all_variants = {
        "BF16": ("BF16", benchmark.run_bf16_fwd, benchmark.run_bf16_bwd, BF16_TOLERANCE),
        "MXFP8": ("MXFP8", benchmark.run_mxfp8_fwd, benchmark.run_mxfp8_bwd, MXFP8_TOLERANCE),
        "BLOCKWISE": ("BLOCKWISE", benchmark.run_blockwise_fwd, benchmark.run_blockwise_bwd, BLOCKWISE_TOLERANCE),
    }
    unknown = [v for v in VARIANTS if v not in all_variants]
    if unknown:
        raise ValueError(f"unknown VARIANTS {unknown}; choose from {sorted(all_variants)}")
    variants = tuple(all_variants[v] for v in VARIANTS)

    reference = run_reference_bf16(*inputs)
    for precision, run_fwd, run_bwd, tolerance in variants:
        check_benchmark_correctness(precision, run_fwd, run_bwd, reference, tolerance, rank)
    del reference

    for precision, run_fwd, run_bwd, _ in variants:
        fwd_ms = benchmark_fwd(run_fwd, device)
        bwd_ms = benchmark_bwd(run_fwd, run_bwd, device)
        if rank == 0:
            fwd_tflops = get_tflops(fwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM)
            bwd_tflops = get_tflops(bwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM, backward=True)
            print(f"{precision}: forward {fwd_ms:.3f} ms, {fwd_tflops:.1f} TFLOP/s; backward {bwd_ms:.3f} ms, {bwd_tflops:.1f} TFLOP/s")

    dist.barrier()
    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
