#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>

using namespace kittens;

namespace mxfp8 {

// Every FP8 recipe in MoK stores E4M3 data plus UE8M0 scales in the tcgen05 block-scaled layout, which carries one
// scale per 32 consecutive K elements. Coarser recipes replicate their scale over the 32-element hardware sub-blocks
// it covers, so the GEMM mainloop is shared by all recipes:
//   K_BLOCK_SIZE = 32:  MXFP8, one scale per 1x32 block
//   K_BLOCK_SIZE = 128: blockwise FP8, one scale per 1x128 block (activations), or per 128x128 block when the caller
//                       supplies the scale (weights)
static constexpr int HW_BLOCK_SIZE = 32;
static constexpr int BLOCKWISE_BLOCK_SIZE = 128;

struct config {
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;

    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = Mb;
};

struct globals {
    using x_bf16_tile = st_bf<config::Mb, config::Nb, false>;
    using x_fp8_tile = st_fp8e4m3<config::Mb, config::Nb, false>;
    using x_sc_tile = st_fp8e8m0<32, 16, false>;

    using x_bf16_gl = gl<bf16, 1, -1, -1, -1, x_bf16_tile>;
    using x_fp8_gl = gl<fp8e4m3, 1, -1, -1, -1, x_fp8_tile>;
    using x_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, x_sc_tile>;

    x_bf16_gl x_bf16;      // (E, M, N) bf16 input (E = 1 for 2D inputs)
    x_fp8_gl x_fp8;        // (E, M, N) MXFP8 output
    x_sc_gl x_sc;          // (E * M // 128, N // 128, 32, 16) fp8e8m0 block scales
    x_fp8_gl x_fp8_t;      // (E, N, M) MXFP8 output, transposed
    x_sc_gl x_sc_t;        // (E * N // 128, M // 128, 32, 16) fp8e8m0 block scales, transposed

    __host__ inline dim3 grid() const {
        return dim3(x_bf16.cols() / config::Nb, x_bf16.rows() / config::Mb, x_bf16.depth());
    }
    __host__ inline int dynamic_shared_memory() const {
        return config::Mb * config::Nb * sizeof(bf16)
             + config::Mb * config::Nb * sizeof(fp8e4m3) + 32 * 16 * sizeof(fp8e8m0)
             + 1024;
    }
};

// Absolute maximum of one 32-element block, kept as a packed pair
static __device__ __forceinline__ bf16_2 block_amax(const bf16_2 (&values_bf16)[16]) {
    bf16_2 amax = __habs2(values_bf16[0]);
    #pragma unroll
    for (int i = 1; i < 16; i++)
        amax = __hmax2(amax, __habs2(values_bf16[i]));
    return amax;
}

static __device__ __forceinline__ float amax_to_float(const bf16_2 amax) {
    return __bfloat162float(__hmax(amax.x, amax.y));
}

// Compute the e8m0 scale, rounding towards positive infinity and saturating to finite (https://arxiv.org/pdf/2506.08027)
static __device__ __forceinline__ uint32_t scale_byte_from_amax(const float amax) {
    const float scale = max(amax * 0.002232142857f, 0.000000000001f);
    uint16_t scale_fp8x2;
    asm volatile("{cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;}" : "=h"(scale_fp8x2) : "f"(scale), "f"(scale));
    return scale_fp8x2 & 0xFF;
}

static __device__ __forceinline__ float scale_inv_from_byte(const uint32_t scale_byte) {
    return __uint_as_float((254u - scale_byte) << 23); // directly build float32 reciprocal without division
}

// Replicates one scale byte over the 4 hardware sub-blocks of a 128-element row
static __device__ __forceinline__ uint32_t replicate_scale_byte(const uint32_t scale_byte) {
    return scale_byte * 0x01010101u;
}

// Quantizes one 32-element block with a precomputed scale
static __device__ __forceinline__ void quantize_block_with_scale(
    const bf16_2 (&values_bf16)[16],
    uint32_t (&values_fp8)[8],
    const float scale_inv
) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const float2 v01_fp32 = __bfloat1622float2(values_bf16[i * 2]);
        const float2 v23_fp32 = __bfloat1622float2(values_bf16[i * 2 + 1]);
        uint16_t v01_fp8, v23_fp8;
        asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(v01_fp8) : "f"(v01_fp32.x * scale_inv), "f"(v01_fp32.y * scale_inv));
        asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(v23_fp8) : "f"(v23_fp32.x * scale_inv), "f"(v23_fp32.y * scale_inv));
        values_fp8[i] = static_cast<uint32_t>(v01_fp8) | (static_cast<uint32_t>(v23_fp8) << 16);
    }
}

// Quantizes one 32-element block with its own scale (MXFP8)
static __device__ __forceinline__ void quantize_single_block(
    const bf16_2 (&values_bf16)[16],
    uint32_t (&values_fp8)[8],
    uint32_t &scale_byte
) {
    scale_byte = scale_byte_from_amax(amax_to_float(block_amax(values_bf16)));
    quantize_block_with_scale(values_bf16, values_fp8, scale_inv_from_byte(scale_byte));
}

static __device__ __forceinline__ void dequantize_single_block(
    const uint32_t (&values_fp8)[8],
    const uint32_t scale_byte,
    float2 (&values_fp32)[16]
) {
    const float scale = __uint_as_float(scale_byte << 23);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint16_t v01_fp8 = static_cast<uint16_t>(values_fp8[i] >> (h * 16));
            const __half2_raw v01_fp16 = __nv_cvt_fp8x2_to_halfraw2(v01_fp8, __NV_E4M3); // no direct conversion from fp8 -> fp32
            const float2 v01_fp32 = __half22float2(*reinterpret_cast<const __half2 *>(&v01_fp16));
            values_fp32[i * 2 + h] = float2{v01_fp32.x * scale, v01_fp32.y * scale};
        }
    }
}

// Absolute maximum of a full 128x128 BF16 tile, computed cooperatively by threads 0-127 (one row each).
// Must be called by all 128 threads of the block; the result is returned to every caller.
static __device__ __forceinline__ float tile_amax(
    const globals::x_bf16_tile &x_bf16_tile,
    float (&scratch)[4],
    const int tid
) {
    constexpr int TILE_SIZE = 128;
    const uint32_t bf16_src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&x_bf16_tile));
    bf16_2 amax = __floats2bfloat162_rn(0.0f, 0.0f);
    #pragma unroll 8
    for (int k = 0; k < TILE_SIZE / 2; k++) {
        const int col = (tid * 4 + k * 2) % TILE_SIZE; // rotate per thread to spread shared memory banks
        bf16_2 value;
        move<bf16_2>::lds(value, bf16_src_addr + (tid * TILE_SIZE + col) * sizeof(bf16));
        amax = __hmax2(amax, __habs2(value));
    }
    float amax_fp32 = amax_to_float(amax);
    #pragma unroll
    for (int offset = WARP_THREADS / 2; offset > 0; offset >>= 1)
        amax_fp32 = fmaxf(amax_fp32, __shfl_xor_sync(0xFFFFFFFF, amax_fp32, offset));
    if (tid % WARP_THREADS == 0)
        scratch[tid / WARP_THREADS] = amax_fp32;
    __syncthreads();
    return fmaxf(fmaxf(scratch[0], scratch[1]), fmaxf(scratch[2], scratch[3]));
}

// Blockwise (1x128) quantization of a 128x128 BF16 tile into the transposed FP8 layout. Each 128-element row of the
// transposed output is one column of the source, so its scale needs the amax over all 128 source rows before any
// element can be quantized, and the source column is strided in shared memory. Rather than one thread per column
// reading 16-bit elements (and either holding the whole column in registers or reading it twice), thread t handles the
// column pair p (source columns 2p and 2p+1, one 32-bit word per source row) for the 64 source rows whose 4-row group
// has parity h: pass 1 accumulates both column amaxes with packed bf16x2 max and exchanges them with the thread that
// holds the other row parity through a 512-byte scratch buffer; pass 2 re-reads the 64 words and emits 4 consecutive
// tokens of each column as one 32-bit word. The lane -> (p, h) assignment makes every load and store conflict-free:
// loads of one warp cover 32 distinct words of a source row (the bank of a word depends on p only, since rows are a
// multiple of 128 bytes apart), stores rotate the row group across lanes and give the two half-warps opposite row
// parities (32 distinct banks of the 128-byte transposed rows).
//   SRC_ROW_STRIDE: row stride of the BF16 source in elements (a multiple of 64)
//   FIXED_SCALE:    use the caller's scale byte for every column (no amax pass, no scratch)
// Must be called by 128 threads (tid 0-127); barrier_id is a named barrier private to these threads.
template <int SRC_ROW_STRIDE, bool FIXED_SCALE>
static __device__ __forceinline__ void quantize_tile_transposed_blockwise(
    const uint32_t bf16_src_addr,
    const uint32_t fp8_dst_addr,
    const uint32_t sc_dst_addr,
    const int tid,
    const int barrier_id,
    const uint32_t fixed_scale_byte,
    const uint32_t amax_scratch_addr
) {
    constexpr int TILE_SIZE = 128;
    static_assert(SRC_ROW_STRIDE % 64 == 0, "rows must be a multiple of 128 bytes apart for conflict-free packed loads");
    const int lane = tid % WARP_THREADS;
    const int warp = tid / WARP_THREADS;
    const int quarter = 2 * (warp % 2) + lane / 16;      // which 16 column pairs this half-warp covers
    const int p = lane % 16 + 16 * quarter;              // column pair: source columns 2p and 2p+1
    const int h = (lane / 16) ^ (warp / 2);              // parity of this thread's 4-row source groups

    auto load_word = [&](int src_row) {
        bf16_2 v;
        move<bf16_2>::lds(v, bf16_src_addr + (src_row * SRC_ROW_STRIDE + 2 * p) * sizeof(bf16));
        return v;
    };

    uint32_t scale_byte0 = fixed_scale_byte, scale_byte1 = fixed_scale_byte;
    if constexpr (!FIXED_SCALE) {
        // Pass 1: column amaxes of this thread's 64 rows, then the other parity's from the partner thread
        bf16_2 amax = __floats2bfloat162_rn(0.0f, 0.0f);
        #pragma unroll 4
        for (int k = 0; k < 16; ++k) {
            const int src_row = (2 * k + h) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                amax = __hmax2(amax, __habs2(load_word(src_row + i)));
        }
        move<bf16_2>::sts(amax_scratch_addr + (h * 64 + p) * sizeof(bf16_2), amax);
        group<TILE_SIZE / WARP_THREADS>::sync(barrier_id);
        bf16_2 other_amax;
        move<bf16_2>::lds(other_amax, amax_scratch_addr + ((h ^ 1) * 64 + p) * sizeof(bf16_2));
        amax = __hmax2(amax, other_amax);
        scale_byte0 = scale_byte_from_amax(__bfloat162float(amax.x));
        scale_byte1 = scale_byte_from_amax(__bfloat162float(amax.y));
    }

    // Pass 2: quantize; 4 consecutive source rows of a column form one 32-bit word of the transposed row
    const float scale_inv0 = scale_inv_from_byte(scale_byte0);
    const float scale_inv1 = scale_inv_from_byte(scale_byte1);
    const uint32_t fp8_dst_addr0 = fp8_dst_addr + (2 * p) * TILE_SIZE; // transposed row 2p
    const uint32_t fp8_dst_addr1 = fp8_dst_addr0 + TILE_SIZE;          // transposed row 2p + 1
    #pragma unroll 2
    for (int s = 0; s < 16; ++s) {
        const int k = (lane % 16 + s) % 16; // rotate the row group across lanes so that the stores hit distinct banks
        const int src_row = (2 * k + h) * 4;
        float2 v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] = __bfloat1622float2(load_word(src_row + i));
        uint16_t c0_01, c0_23, c1_01, c1_23;
        asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(c0_01) : "f"(v[0].x * scale_inv0), "f"(v[1].x * scale_inv0));
        asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(c0_23) : "f"(v[2].x * scale_inv0), "f"(v[3].x * scale_inv0));
        asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(c1_01) : "f"(v[0].y * scale_inv1), "f"(v[1].y * scale_inv1));
        asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(c1_23) : "f"(v[2].y * scale_inv1), "f"(v[3].y * scale_inv1));
        move<int>::sts(fp8_dst_addr0 + src_row, std::bit_cast<int>(static_cast<uint32_t>(c0_01) | (static_cast<uint32_t>(c0_23) << 16)));
        move<int>::sts(fp8_dst_addr1 + src_row, std::bit_cast<int>(static_cast<uint32_t>(c1_01) | (static_cast<uint32_t>(c1_23) << 16)));
    }

    // Scale words of transposed rows 2p and 2p+1 (tcgen05 scale-factor layout, see quantize_tile)
    const int row0 = 2 * p, row1 = 2 * p + 1;
    move<int>::sts(sc_dst_addr + (row0 % 32) * 16 + (row0 / 32) * 4, std::bit_cast<int>(replicate_scale_byte(scale_byte0)));
    move<int>::sts(sc_dst_addr + (row1 % 32) * 16 + (row1 / 32) * 4, std::bit_cast<int>(replicate_scale_byte(scale_byte1)));
}

// Quantizes a 128x128 BF16 tile held in shared memory into FP8 data and scale tiles, in the normal and/or transposed layouts.
//   K_BLOCK_SIZE = 32:  one scale per 32-element sub-block of each row (MXFP8)
//   K_BLOCK_SIZE = 128: one scale per 128-element row, replicated over its 4 sub-blocks (blockwise FP8)
//   FIXED_SCALE:        use the caller's scale byte for every row (e.g. one scale for the whole 128x128 tile)
//   SPLIT_PASSES:       threads 0-127 produce the transposed outputs while threads 128-255 produce the normal ones
//   SCALE_ROWS:         multiply row r by row_scales[r] (rounded to BF16) before quantizing
//   amax_scratch_addr:  512 bytes of shared memory (blockwise FP8 with a transposed output and no fixed scale only)
// Named barriers barrier_id .. barrier_id + 6 are used by the threads that call this.
template <bool RETURN_NORMAL, bool RETURN_TRANSPOSED, int SRC_ROW_STRIDE = 128, bool SPLIT_PASSES = false, bool SCALE_ROWS = false,
          int K_BLOCK_SIZE = HW_BLOCK_SIZE, bool FIXED_SCALE = false>
static __device__ __forceinline__ void quantize_tile(
    const globals::x_bf16_tile &x_bf16_tile,
    const globals::x_fp8_tile &x_fp8_tile,
    const globals::x_sc_tile &x_sc_tile,
    const globals::x_fp8_tile &x_fp8_t_tile,
    const globals::x_sc_tile &x_sc_t_tile,
    const float *row_scales,
    const int _tid,
    const int barrier_id,
    const uint32_t fixed_scale_byte = 0,
    const uint32_t amax_scratch_addr = 0
) {
    constexpr int TILE_SIZE = 128;
    static_assert(RETURN_NORMAL || RETURN_TRANSPOSED, "At least one output pair must be requested");
    static_assert(!SPLIT_PASSES || (RETURN_NORMAL && RETURN_TRANSPOSED), "SPLIT_PASSES requires both outputs");
    static_assert(K_BLOCK_SIZE == HW_BLOCK_SIZE || K_BLOCK_SIZE == BLOCKWISE_BLOCK_SIZE, "K_BLOCK_SIZE must be 32 (MXFP8) or 128 (blockwise FP8)");
    static_assert(!FIXED_SCALE || K_BLOCK_SIZE == BLOCKWISE_BLOCK_SIZE, "A fixed scale implies one scale per row");
    constexpr bool PER_SUB_BLOCK_SCALE = K_BLOCK_SIZE == HW_BLOCK_SIZE;
    // Blockwise FP8 with row scaling: the normal-layout threads, which load and scale whole rows anyway, write the scaled
    // rows back over the source before the transposed-layout threads read it, so the transposed pass needs no scaling.
    constexpr bool TRANSPOSED_READS_PRESCALED = !PER_SUB_BLOCK_SCALE && SCALE_ROWS && RETURN_TRANSPOSED;
    static_assert(!TRANSPOSED_READS_PRESCALED || SPLIT_PASSES, "Blockwise FP8 row scaling with a transposed output requires SPLIT_PASSES");

    // Excess threads have no tile rows to handle. Caller must ensure that this is called by threads 0-127, or 0-255 with SPLIT_PASSES
    const int tid = SPLIT_PASSES ? _tid % TILE_SIZE : _tid;
    if (!SPLIT_PASSES && tid >= TILE_SIZE) return;

    constexpr int NUM_K_BLOCKS = TILE_SIZE / HW_BLOCK_SIZE; // 4 hardware sub-blocks per row
    constexpr int PACKED_PER_K_BLOCK = HW_BLOCK_SIZE / 2;  // 16

    const uint32_t bf16_src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&x_bf16_tile));

    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        const bool transposed = pass == 0;
        if ((transposed && !RETURN_TRANSPOSED) || (!transposed && !RETURN_NORMAL)) continue;
        if (SPLIT_PASSES && transposed != (_tid < TILE_SIZE)) continue;
        const uint32_t fp8_dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(transposed ? &x_fp8_t_tile : &x_fp8_tile));
        const uint32_t sc_dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(transposed ? &x_sc_t_tile : &x_sc_tile));

        if constexpr (!PER_SUB_BLOCK_SCALE) {
            if (transposed) {
                if constexpr (TRANSPOSED_READS_PRESCALED)
                    group<2 * TILE_SIZE / WARP_THREADS>::sync(barrier_id + 6); // scaled rows written back by the normal-layout threads
                quantize_tile_transposed_blockwise<SRC_ROW_STRIDE, FIXED_SCALE>(
                    bf16_src_addr, fp8_dst_addr, sc_dst_addr, tid, barrier_id + 4, fixed_scale_byte, amax_scratch_addr);
                continue;
            }
        }
        const int row = transposed ? (tid % 64) * 2 + tid / 64 : tid;

        auto load_k_block = [&](int k_block_idx, bf16_2 (&dst)[PACKED_PER_K_BLOCK]) {
            // Load one 32-element block of this thread's row from shared memory w/ custom swizzling
            #pragma unroll
            for (int k = 0; k < PACKED_PER_K_BLOCK; k++) {
                const int col = k_block_idx*HW_BLOCK_SIZE + (tid*4 + k*2) % HW_BLOCK_SIZE;
                if (transposed) {
                    move<bf16>::lds(dst[k].x, bf16_src_addr + (col*SRC_ROW_STRIDE + row) * sizeof(bf16));
                    move<bf16>::lds(dst[k].y, bf16_src_addr + ((col+1)*SRC_ROW_STRIDE + row) * sizeof(bf16));
                    if constexpr (SCALE_ROWS) {
                        const float2 val = __bfloat1622float2(dst[k]);
                        dst[k] = __float22bfloat162_rn(float2{val.x * row_scales[col], val.y * row_scales[col + 1]});
                    }
                } else {
                    move<bf16_2>::lds(dst[k], bf16_src_addr + (row*SRC_ROW_STRIDE + col) * sizeof(bf16));
                    if constexpr (SCALE_ROWS) {
                        const float2 val = __bfloat1622float2(dst[k]);
                        dst[k] = __float22bfloat162_rn(float2{val.x * row_scales[row], val.y * row_scales[row]});
                    }
                }
            }
        };
        auto store_k_block = [&](int k_block_idx, const uint32_t (&x_fp8_reg)[PACKED_PER_K_BLOCK / 2]) {
            // Store one quantized 32-element block to shared memory
            #pragma unroll
            for (int k = 0; k < PACKED_PER_K_BLOCK / 2; k++) {
                const int col = k_block_idx*HW_BLOCK_SIZE + (tid*4 + k*4) % HW_BLOCK_SIZE;
                move<int>::sts(fp8_dst_addr + row*TILE_SIZE + col, std::bit_cast<int>(x_fp8_reg[k]));
            }
        };

        uint32_t scale_word = 0;
        if constexpr (PER_SUB_BLOCK_SCALE) {
            auto quantize_k_block = [&](int k_block_idx, const bf16_2 (&src)[PACKED_PER_K_BLOCK]) {
                // Quantize one 32-element block with its own scale and store the FP8 output to shared memory
                uint32_t x_fp8_reg[PACKED_PER_K_BLOCK / 2];
                uint32_t scale_byte;
                quantize_single_block(src, x_fp8_reg, scale_byte);
                scale_word |= scale_byte << (k_block_idx * 8);
                store_k_block(k_block_idx, x_fp8_reg);
            };

            if (!transposed) {
                // This writes FP8 output in place over the BF16 source, so it must load the whole tile up front
                bf16_2 x_bf16_reg[NUM_K_BLOCKS][PACKED_PER_K_BLOCK];
                #pragma unroll
                for (int j = 0; j < NUM_K_BLOCKS; j++)
                    load_k_block((j + tid/8) % NUM_K_BLOCKS, x_bf16_reg[j]);
                group<TILE_SIZE / WARP_THREADS>::sync(barrier_id); // in-place writes may begin
                #pragma unroll
                for (int j = 0; j < NUM_K_BLOCKS; j++)
                    quantize_k_block((j + tid/8) % NUM_K_BLOCKS, x_bf16_reg[j]);
            } else {
                #pragma unroll 1 // otherwise we get hundreds of register spills
                for (int j = 0; j < NUM_K_BLOCKS; j++) {
                    const int k_block_idx = (j + tid/8) % NUM_K_BLOCKS;
                    bf16_2 x_bf16_reg[PACKED_PER_K_BLOCK];
                    load_k_block(k_block_idx, x_bf16_reg);
                    quantize_k_block(k_block_idx, x_bf16_reg);
                }
            }
        } else {
            // Normal layout, one scale per 128-element row: load the whole row up front (the FP8 output may alias the
            // BF16 source), take its amax (or the caller's scale), then quantize
            uint32_t scale_byte = fixed_scale_byte;
            bf16_2 x_bf16_reg[NUM_K_BLOCKS][PACKED_PER_K_BLOCK];
            #pragma unroll
            for (int j = 0; j < NUM_K_BLOCKS; j++)
                load_k_block((j + tid/8) % NUM_K_BLOCKS, x_bf16_reg[j]);
            if constexpr (TRANSPOSED_READS_PRESCALED) {
                // Publish the scaled row (in place) for the transposed-layout threads
                #pragma unroll
                for (int j = 0; j < NUM_K_BLOCKS; j++) {
                    const int k_block_idx = (j + tid/8) % NUM_K_BLOCKS;
                    #pragma unroll
                    for (int k = 0; k < PACKED_PER_K_BLOCK; k++) {
                        const int col = k_block_idx*HW_BLOCK_SIZE + (tid*4 + k*2) % HW_BLOCK_SIZE;
                        move<bf16_2>::sts(bf16_src_addr + (row*SRC_ROW_STRIDE + col) * sizeof(bf16), x_bf16_reg[j][k]);
                    }
                }
                group<2 * TILE_SIZE / WARP_THREADS>::sync(barrier_id + 6);
            }
            if constexpr (!FIXED_SCALE) {
                bf16_2 amax = block_amax(x_bf16_reg[0]);
                #pragma unroll
                for (int j = 1; j < NUM_K_BLOCKS; j++)
                    amax = __hmax2(amax, block_amax(x_bf16_reg[j]));
                scale_byte = scale_byte_from_amax(amax_to_float(amax));
            }
            group<TILE_SIZE / WARP_THREADS>::sync(barrier_id); // in-place writes may begin
            const float scale_inv = scale_inv_from_byte(scale_byte);
            #pragma unroll
            for (int j = 0; j < NUM_K_BLOCKS; j++) {
                uint32_t x_fp8_reg[PACKED_PER_K_BLOCK / 2];
                quantize_block_with_scale(x_bf16_reg[j], x_fp8_reg, scale_inv);
                store_k_block((j + tid/8) % NUM_K_BLOCKS, x_fp8_reg);
            }
            scale_word = replicate_scale_byte(scale_byte);
        }

        // Store the scales to shared memory. Each thread will access 1 bank, so no need to swizzle,
        // but we do have to follow this complicated layout pattern made by NVIDIA:
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
        move<int>::sts(sc_dst_addr + (row % 32) * 16 + (row / 32) * 4, std::bit_cast<int>(scale_word));
    }
}

// Standalone quantization kernel (used to prequantize weights).
//   K_BLOCK_SIZE = 32,  BLOCK_ROWS = 1:   MXFP8 (1x32 blocks)
//   K_BLOCK_SIZE = 128, BLOCK_ROWS = 1:   blockwise FP8 activations (1x128 blocks)
//   K_BLOCK_SIZE = 128, BLOCK_ROWS = 128: blockwise FP8 weights (128x128 blocks)
template <bool RETURN_NORMAL, bool RETURN_TRANSPOSED, int K_BLOCK_SIZE = HW_BLOCK_SIZE, int BLOCK_ROWS = 1>
static __device__ __forceinline__ void quantize_kernel(const globals &G) {
    static_assert(BLOCK_ROWS == 1 || BLOCK_ROWS == BLOCKWISE_BLOCK_SIZE, "BLOCK_ROWS must be 1 or 128");
    static_assert(BLOCK_ROWS == 1 || K_BLOCK_SIZE == BLOCKWISE_BLOCK_SIZE, "128-row blocks require 128-wide blocks");

    // Allocate shared memory
    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);
    auto &x_bf16_tile = *reinterpret_cast<globals::x_bf16_tile *>(smem_base_addr);
    auto &x_fp8_tile = *reinterpret_cast<globals::x_fp8_tile *>(&x_bf16_tile);
    auto &x_sc_tile = *reinterpret_cast<globals::x_sc_tile *>(reinterpret_cast<uint64_t>(&x_bf16_tile) + sizeof(globals::x_fp8_tile));
    auto &x_fp8_t_tile = *reinterpret_cast<globals::x_fp8_tile *>(reinterpret_cast<uint64_t>(&x_bf16_tile) + sizeof(globals::x_bf16_tile));
    auto &x_sc_t_tile = *reinterpret_cast<globals::x_sc_tile *>(reinterpret_cast<uint64_t>(&x_fp8_t_tile) + sizeof(globals::x_fp8_tile));

    // Calculate indices
    const int tid = threadIdx.x;
    const int expert = blockIdx.z;
    const int row = blockIdx.y;
    const int col = blockIdx.x;

    // Initialize mbarrier and initiate TMA load
    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect(inputs_arrived, x_bf16_tile);
        tma::load_async(x_bf16_tile, G.x_bf16, {expert, row, col}, inputs_arrived);
    }

    // Wait for the TMA load to complete
    __syncthreads();
    wait(inputs_arrived, 0);

    // Quantize
    if constexpr (BLOCK_ROWS == BLOCKWISE_BLOCK_SIZE) {
        // One scale for the whole 128x128 tile, shared by the normal and transposed outputs
        __shared__ float amax_scratch[4];
        const uint32_t scale_byte = scale_byte_from_amax(tile_amax(x_bf16_tile, amax_scratch, tid));
        quantize_tile<RETURN_NORMAL, RETURN_TRANSPOSED, 128, false, false, K_BLOCK_SIZE, true>(
            x_bf16_tile, x_fp8_tile, x_sc_tile, x_fp8_t_tile, x_sc_t_tile, nullptr, tid, 1, scale_byte);
    } else {
        __shared__ __align__(16) uint32_t column_amax_scratch[128]; // blockwise transposed pass only
        quantize_tile<RETURN_NORMAL, RETURN_TRANSPOSED, 128, false, false, K_BLOCK_SIZE>(
            x_bf16_tile, x_fp8_tile, x_sc_tile, x_fp8_t_tile, x_sc_t_tile, nullptr, tid, 1, 0,
            static_cast<uint32_t>(__cvta_generic_to_shared(column_amax_scratch)));
    }
    __syncthreads();

    // Store to global memory
    if (tid == 0) {
        if constexpr (RETURN_TRANSPOSED) {
            tma::store_async(G.x_fp8_t, x_fp8_t_tile, {expert, col, row});
            tma::store_async(G.x_sc_t, x_sc_t_tile, {expert * static_cast<int>(gridDim.x) + col, row, 0, 0});
        }
        if constexpr (RETURN_NORMAL) {
            tma::store_async(G.x_fp8, x_fp8_tile, {expert, row, col});
            tma::store_async(G.x_sc, x_sc_tile, {expert * static_cast<int>(gridDim.y) + row, col, 0, 0});
        }
    }
}

template <int K_BLOCK_SIZE, int BLOCK_ROWS>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
quantize_entrypoint_impl(
    const at::Tensor &x_bf16,
    const bool return_normal,
    const bool return_transposed
) {
    using C = mxfp8::config;
    using G = mxfp8::globals;

    const int E = x_bf16.dim() == 3 ? static_cast<int>(x_bf16.size(0)) : 1;
    const int M = static_cast<int>(x_bf16.size(-2));
    const int N = static_cast<int>(x_bf16.size(-1));

    at::Tensor x_fp8, x_sc, x_fp8_t, x_sc_t;
    if (return_normal) {
        x_fp8 = at::empty(x_bf16.sizes(), x_bf16.options().dtype(at::kFloat8_e4m3fn));
        x_sc = at::empty({E * (M / 128), N / 128, 32, 16}, x_bf16.options().dtype(at::kByte));
    }
    if (return_transposed) {
        x_fp8_t = x_bf16.dim() == 3 ? at::empty({E, N, M}, x_bf16.options().dtype(at::kFloat8_e4m3fn))
                                    : at::empty({N, M}, x_bf16.options().dtype(at::kFloat8_e4m3fn));
        x_sc_t = at::empty({E * (N / 128), M / 128, 32, 16}, x_bf16.options().dtype(at::kByte));
    }

    G g {
        .x_bf16 = kittens::py::tensor_to_gl<G::x_bf16_gl>(x_bf16),
        .x_fp8 = return_normal ? kittens::py::tensor_to_gl<G::x_fp8_gl>(x_fp8)
                               : kittens::py::make_fake_gl<G::x_fp8_gl>(1, E, M, N),
        .x_sc = return_normal ? kittens::py::tensor_to_gl<G::x_sc_gl>(x_sc)
                              : kittens::py::make_fake_gl<G::x_sc_gl>(E * (M / 128), N / 128, 32, 16),
        .x_fp8_t = return_transposed ? kittens::py::tensor_to_gl<G::x_fp8_gl>(x_fp8_t)
                                     : kittens::py::make_fake_gl<G::x_fp8_gl>(1, E, N, M),
        .x_sc_t = return_transposed ? kittens::py::tensor_to_gl<G::x_sc_gl>(x_sc_t)
                                    : kittens::py::make_fake_gl<G::x_sc_gl>(E * (N / 128), M / 128, 32, 16)
    };

    if (return_normal && return_transposed)
        kittens::py::launch_kernel<C, G, mxfp8::quantize_kernel<true, true, K_BLOCK_SIZE, BLOCK_ROWS>>(g);
    else if (return_normal)
        kittens::py::launch_kernel<C, G, mxfp8::quantize_kernel<true, false, K_BLOCK_SIZE, BLOCK_ROWS>>(g);
    else
        kittens::py::launch_kernel<C, G, mxfp8::quantize_kernel<false, true, K_BLOCK_SIZE, BLOCK_ROWS>>(g);

    return {x_fp8, x_sc, x_fp8_t, x_sc_t};
}

// MXFP8: one scale per 1x32 block
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
quantize_entrypoint(
    const at::Tensor &x_bf16,
    const bool return_normal,
    const bool return_transposed
) {
    return quantize_entrypoint_impl<HW_BLOCK_SIZE, 1>(x_bf16, return_normal, return_transposed);
}

// Blockwise FP8: one scale per 1x128 block (block_rows = 1, activations) or per 128x128 block (block_rows = 128, weights)
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
blockwise_quantize_entrypoint(
    const at::Tensor &x_bf16,
    const bool return_normal,
    const bool return_transposed,
    const int block_rows
) {
    switch (block_rows) {
        case 1:
            return quantize_entrypoint_impl<BLOCKWISE_BLOCK_SIZE, 1>(x_bf16, return_normal, return_transposed);
        case BLOCKWISE_BLOCK_SIZE:
            return quantize_entrypoint_impl<BLOCKWISE_BLOCK_SIZE, BLOCKWISE_BLOCK_SIZE>(x_bf16, return_normal, return_transposed);
        default:
            throw std::runtime_error("MoK: blockwise_fp8_quantize block_rows must be 1 or 128, got " + std::to_string(block_rows));
    }
}

} // namespace mxfp8
