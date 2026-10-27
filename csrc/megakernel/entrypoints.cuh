#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "megakernel.cuh"

// MoK supports expert-parallel sizes of 1, 4, 8, 16, 32 and 64 devices. Define MOK_SINGLE_NODE to compile only the
// single-node sizes (1, 4, 8), which roughly halves the build time during development.
#ifdef MOK_SINGLE_NODE
static constexpr const char *MOK_SUPPORTED_DEVICE_COUNTS = "1, 4, 8";
#else
static constexpr const char *MOK_SUPPORTED_DEVICE_COUNTS = "1, 4, 8, 16, 32, 64";
#endif

static constexpr const char *fp8_recipe_name(RoutedPrecision precision) {
    switch (precision) {
        case RoutedPrecision::MXFP8: return "mxfp8";
        case RoutedPrecision::BLOCKWISE_FP8: return "blockwise_fp8";
        default: return "bf16";
    }
}

// FP8 forward: MXFP8 (PRECISION = MXFP8) and blockwise FP8 (PRECISION = BLOCKWISE_FP8) share this entrypoint
template <RoutedPrecision PRECISION>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_fwd_fp8_entrypoint(
    // Inputs and communication buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,

    // Weights
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_routed_gate_sc,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_routed_up_sc,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &w_routed_down_sc,

    // Dispatch/combine schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    static_assert(PRECISION != RoutedPrecision::BF16, "use the BF16 entrypoint");
    const int num_devices = static_cast<int>(x_ptrs.size());

#define MOK_FWD_FP8_CASE(N)                                                                                   \
        case N:                                                                                               \
            return dispatch_mlp_swiglu_combiner<N, PRECISION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(         \
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,                                               \
                w_shared_gate, w_routed_gate, w_routed_gate_sc,                                               \
                w_shared_up, w_routed_up, w_routed_up_sc,                                                     \
                w_shared_down, w_routed_down, w_routed_down_sc,                                               \
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,                   \
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
    switch (num_devices) {
        MOK_FWD_FP8_CASE(1)
        MOK_FWD_FP8_CASE(4)
        MOK_FWD_FP8_CASE(8)
#ifndef MOK_SINGLE_NODE
        MOK_FWD_FP8_CASE(16)
        MOK_FWD_FP8_CASE(32)
        MOK_FWD_FP8_CASE(64)
#endif
        default:
            throw std::runtime_error(std::string("MoK: dispatch_mlp_swiglu_combine_fwd_") + fp8_recipe_name(PRECISION) +
                                     " unsupported num_devices=" + std::to_string(num_devices) +
                                     " (supported: " + MOK_SUPPORTED_DEVICE_COUNTS + ")");
    }
#undef MOK_FWD_FP8_CASE
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_fwd_bf16_entrypoint(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());

#define MOK_FWD_BF16_CASE(N)                                                                                  \
        case N:                                                                                               \
            return dispatch_mlp_swiglu_combiner<N, RoutedPrecision::BF16>::dispatch_mlp_swiglu_combine_fwd_bf16( \
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,                                               \
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,         \
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,                   \
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
    switch (num_devices) {
        MOK_FWD_BF16_CASE(1)
        MOK_FWD_BF16_CASE(4)
        MOK_FWD_BF16_CASE(8)
#ifndef MOK_SINGLE_NODE
        MOK_FWD_BF16_CASE(16)
        MOK_FWD_BF16_CASE(32)
        MOK_FWD_BF16_CASE(64)
#endif
        default:
            throw std::runtime_error("MoK: dispatch_mlp_swiglu_combine_fwd_bf16 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: " + MOK_SUPPORTED_DEVICE_COUNTS + ")");
    }
#undef MOK_FWD_BF16_CASE
}

// FP8 backward: MXFP8 (PRECISION = MXFP8) and blockwise FP8 (PRECISION = BLOCKWISE_FP8) share this entrypoint
template <RoutedPrecision PRECISION>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_bwd_fp8_entrypoint(
    // Symmetric buffers (input/output gradients and router weights)
    const at::Tensor &d_y_buffer,
    const std::vector<int64_t> &d_y_buffer_ptrs,
    const at::Tensor &d_y_shared,
    const at::Tensor &d_x_routed_buffer,
    const std::vector<int64_t> &d_x_routed_buffer_ptrs,
    const at::Tensor &router_weight_buffer,
    const std::vector<int64_t> &router_weight_buffer_ptrs,
    const at::Tensor &d_router_weight_buffer,
    const std::vector<int64_t> &d_router_weight_buffer_ptrs,

    // Weights (routed transposes pre-quantized to FP8)
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate_T,
    const at::Tensor &w_routed_gate_T_sc,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up_T,
    const at::Tensor &w_routed_up_T_sc,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down_T,
    const at::Tensor &w_routed_down_T_sc,

    // Activations saved from the forward
    const at::Tensor &x_fp8_t_routed,
    const at::Tensor &x_sc_t_routed,
    const at::Tensor &gate_shared,
    const at::Tensor &gate_fp8_routed,
    const at::Tensor &gate_sc_routed,
    const at::Tensor &up_shared,
    const at::Tensor &up_fp8_routed,
    const at::Tensor &up_sc_routed,
    const at::Tensor &hidden_shared,
    const at::Tensor &hidden_fp8_t_routed,
    const at::Tensor &hidden_sc_t_routed,

    // Activations and weights for forward replay
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_routed_gate_sc,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_routed_up_sc,

    // Dispatch/combine schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    static_assert(PRECISION != RoutedPrecision::BF16, "use the BF16 entrypoint");
    const int num_devices = static_cast<int>(x_ptrs.size());

#define MOK_BWD_FP8_CASE(N)                                                                                   \
        case N:                                                                                               \
            return dispatch_mlp_swiglu_combiner<N, PRECISION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(         \
                d_y_buffer, d_y_buffer_ptrs, d_y_shared, d_x_routed_buffer, d_x_routed_buffer_ptrs,           \
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs, \
                w_shared_gate, w_routed_gate_T, w_routed_gate_T_sc,                                           \
                w_shared_up, w_routed_up_T, w_routed_up_T_sc,                                                 \
                w_shared_down, w_routed_down_T, w_routed_down_T_sc,                                           \
                x_fp8_t_routed, x_sc_t_routed,                                                                \
                gate_shared, gate_fp8_routed, gate_sc_routed,                                                 \
                up_shared, up_fp8_routed, up_sc_routed,                                                       \
                hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,                                       \
                x, x_ptrs,                                                                                    \
                w_routed_gate, w_routed_gate_sc,                                                              \
                w_routed_up, w_routed_up_sc,                                                                  \
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,                   \
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
    switch (num_devices) {
        MOK_BWD_FP8_CASE(1)
        MOK_BWD_FP8_CASE(4)
        MOK_BWD_FP8_CASE(8)
#ifndef MOK_SINGLE_NODE
        MOK_BWD_FP8_CASE(16)
        MOK_BWD_FP8_CASE(32)
        MOK_BWD_FP8_CASE(64)
#endif
        default:
            throw std::runtime_error(std::string("MoK: dispatch_mlp_swiglu_combine_bwd_") + fp8_recipe_name(PRECISION) +
                                     " unsupported num_devices=" + std::to_string(num_devices) +
                                     " (supported: " + MOK_SUPPORTED_DEVICE_COUNTS + ")");
    }
#undef MOK_BWD_FP8_CASE
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_bwd_bf16_entrypoint(
    const at::Tensor &d_y_buffer,
    const std::vector<int64_t> &d_y_buffer_ptrs,
    const at::Tensor &d_y_shared,
    const at::Tensor &d_x_routed_buffer,
    const std::vector<int64_t> &d_x_routed_buffer_ptrs,
    const at::Tensor &router_weight_buffer,
    const std::vector<int64_t> &router_weight_buffer_ptrs,
    const at::Tensor &d_router_weight_buffer,
    const std::vector<int64_t> &d_router_weight_buffer_ptrs,
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &x_routed,
    const at::Tensor &gate_shared,
    const at::Tensor &gate_routed,
    const at::Tensor &up_shared,
    const at::Tensor &up_routed,
    const at::Tensor &hidden_shared,
    const at::Tensor &hidden_routed,
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());

#define MOK_BWD_BF16_CASE(N)                                                                                  \
        case N:                                                                                               \
            return dispatch_mlp_swiglu_combiner<N, RoutedPrecision::BF16>::dispatch_mlp_swiglu_combine_bwd_bf16( \
                d_y_buffer, d_y_buffer_ptrs, d_y_shared, d_x_routed_buffer, d_x_routed_buffer_ptrs,           \
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs, \
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,         \
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs, \
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,                   \
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
    switch (num_devices) {
        MOK_BWD_BF16_CASE(1)
        MOK_BWD_BF16_CASE(4)
        MOK_BWD_BF16_CASE(8)
#ifndef MOK_SINGLE_NODE
        MOK_BWD_BF16_CASE(16)
        MOK_BWD_BF16_CASE(32)
        MOK_BWD_BF16_CASE(64)
#endif
        default:
            throw std::runtime_error("MoK: dispatch_mlp_swiglu_combine_bwd_bf16 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: " + MOK_SUPPORTED_DEVICE_COUNTS + ")");
    }
#undef MOK_BWD_BF16_CASE
}

// FP8 forward-context recompute: MXFP8 (PRECISION = MXFP8) and blockwise FP8 (PRECISION = BLOCKWISE_FP8) share this entrypoint
template <RoutedPrecision PRECISION>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
recompute_forward_context_fp8_entrypoint(
    // Input buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,

    // Weights
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_routed_gate_sc,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_routed_up_sc,

    // Dispatch schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    static_assert(PRECISION != RoutedPrecision::BF16, "use the BF16 entrypoint");
    const int num_devices = static_cast<int>(x_ptrs.size());

#define MOK_RECOMPUTE_FP8_CASE(N)                                                                             \
        case N:                                                                                               \
            return dispatch_mlp_swiglu_combiner<N, PRECISION>::recompute_forward_context_mxfp8(               \
                x, x_ptrs,                                                                                    \
                w_shared_gate, w_routed_gate, w_routed_gate_sc,                                               \
                w_shared_up, w_routed_up, w_routed_up_sc,                                                     \
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,                   \
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
    switch (num_devices) {
        MOK_RECOMPUTE_FP8_CASE(1)
        MOK_RECOMPUTE_FP8_CASE(4)
        MOK_RECOMPUTE_FP8_CASE(8)
#ifndef MOK_SINGLE_NODE
        MOK_RECOMPUTE_FP8_CASE(16)
        MOK_RECOMPUTE_FP8_CASE(32)
        MOK_RECOMPUTE_FP8_CASE(64)
#endif
        default:
            throw std::runtime_error(std::string("MoK: recompute_forward_context_") + fp8_recipe_name(PRECISION) +
                                     " unsupported num_devices=" + std::to_string(num_devices) +
                                     " (supported: " + MOK_SUPPORTED_DEVICE_COUNTS + ")");
    }
#undef MOK_RECOMPUTE_FP8_CASE
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor>
recompute_forward_context_bf16_entrypoint(
    // Input buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,

    // Weights
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,

    // Dispatch schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());

#define MOK_RECOMPUTE_BF16_CASE(N)                                                                            \
        case N:                                                                                               \
            return dispatch_mlp_swiglu_combiner<N, RoutedPrecision::BF16>::recompute_forward_context_bf16(    \
                x, x_ptrs,                                                                                    \
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,                                       \
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,                   \
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
    switch (num_devices) {
        MOK_RECOMPUTE_BF16_CASE(1)
        MOK_RECOMPUTE_BF16_CASE(4)
        MOK_RECOMPUTE_BF16_CASE(8)
#ifndef MOK_SINGLE_NODE
        MOK_RECOMPUTE_BF16_CASE(16)
        MOK_RECOMPUTE_BF16_CASE(32)
        MOK_RECOMPUTE_BF16_CASE(64)
#endif
        default:
            throw std::runtime_error("MoK: recompute_forward_context_bf16 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: " + MOK_SUPPORTED_DEVICE_COUNTS + ")");
    }
#undef MOK_RECOMPUTE_BF16_CASE
}
