#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "../mxfp8.cuh"
#include "../utils.cuh"
#include "profile.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>

using namespace kittens;

enum class RoutedPrecision {
    BF16,
    MXFP8,         // FP8 E4M3 with one UE8M0 scale per 1x32 block
    BLOCKWISE_FP8, // FP8 E4M3 with one UE8M0 scale per 1x128 block (activations) / 128x128 block (weights)
};

template <int NUM_DEVICES, RoutedPrecision ROUTED_PRECISION = RoutedPrecision::MXFP8>
struct dispatch_mlp_swiglu_combiner {

// Both FP8 recipes share the MXFP8 tensor layouts, scale layouts and GEMM mainloop (blockwise scales are replicated over
// the 32-element hardware sub-blocks they cover); they only differ in the granularity at which scales are computed.
static constexpr bool USE_MXFP8 = ROUTED_PRECISION != RoutedPrecision::BF16;
static constexpr int QUANT_BLOCK_SIZE = ROUTED_PRECISION == RoutedPrecision::BLOCKWISE_FP8 ? mxfp8::BLOCKWISE_BLOCK_SIZE : mxfp8::HW_BLOCK_SIZE;

#include "types.cuh"

#include "utils.cuh"
#include "dispatch_combine.cuh"
#include "swiglu.cuh"
#include "grouped_gemm.cuh"

#include "forward.cuh"
#include "backward.cuh"
#include "recompute_forward_context.cuh"

}; // struct dispatch_mlp_swiglu_combiner
