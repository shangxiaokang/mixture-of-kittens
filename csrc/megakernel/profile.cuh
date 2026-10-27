#pragma once

// Optional per-stage timing of the megakernels, enabled with -DMOK_PROFILE (e.g. EXTRA_NVCCFLAGS="-DMOK_SINGLE_NODE -DMOK_PROFILE").
// Thread 0 of every CTA accumulates, per stage, the wall-clock time (%globaltimer, ns) it spent in each task, the task
// count and the earliest start / latest end into a device-global table that the host reads through
// mok.ops._C.profile_read() and clears with mok.ops._C.profile_reset(). Without the macro the hooks compile to nothing.
namespace mok_profile {

enum stage : int {
    // Forward megakernel (and the recompute-forward-context megakernel). The backward's replay of these stages for
    // macrobatches > 1 is recorded under the B_REPLAY_* stages instead.
    F_DISPATCH = 0,        // dispatch task (TMA pull + quantize + store)
    F_DISPATCH_QUANT,      // quantize phase of the dispatch task (FP8 recipes only)
    F_COMBINE,
    F_GATE, F_UP,          // routed gate / up GEMM tasks
    F_SWIGLU,              // routed SwiGLU task
    F_SWIGLU_COMPUTE,      // elementwise phase of a routed SwiGLU tile (FP8 recipes only)
    F_SWIGLU_QUANT,        // quantize phase of a routed SwiGLU tile (FP8 recipes only)
    F_DOWN,                // routed down GEMM tasks
    F_SHARED,              // all shared-expert tasks
    // Backward megakernel
    B_REV_COMBINE = 16,    // reverse combine (dispatch of d_y with router-weight scaling + quantize)
    B_REV_COMBINE_QUANT,   // quantize phase of the reverse combine (FP8 recipes only)
    B_REV_DISPATCH,        // reverse dispatch (combine of d_x)
    B_PRELOAD,             // router weight preload
    B_REPLAY_DISPATCH,     // replayed dispatch of x (macrobatches > 1)
    B_DGRAD_DOWN,
    B_SWIGLU_BWD,
    B_SWIGLU_BWD_COMPUTE,  // SwiGLU backward + normal-layout quantize of a routed tile (FP8 recipes only)
    B_SWIGLU_BWD_TQUANT,   // transposed-layout quantize of a routed tile (FP8 recipes only)
    B_DGRAD_GATE_UP,
    B_WGRAD_DOWN, B_WGRAD_GATE, B_WGRAD_UP,
    B_REPLAY_GATE, B_REPLAY_UP, B_REPLAY_SWIGLU,
    B_SHARED,              // all shared-expert tasks
    B_REPLAY_DISPATCH_QUANT,  // quantize phase of a replayed dispatch task (FP8 recipes only)
    B_REPLAY_SWIGLU_COMPUTE,  // elementwise phase of a replayed routed SwiGLU tile (FP8 recipes only)
    B_REPLAY_SWIGLU_QUANT,    // quantize phase of a replayed routed SwiGLU tile (FP8 recipes only)
    NUM_STAGES = 40
};
static constexpr int NUM_FIELDS = 4; // sum of durations (ns), count, min start (ns), max end (ns)

#ifdef MOK_PROFILE
__device__ unsigned long long table[NUM_STAGES * NUM_FIELDS];

static __device__ __forceinline__ unsigned long long now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __device__ __forceinline__ void record(const int stage, const unsigned long long start) {
    if (threadIdx.x != 0) return;
    const unsigned long long end = now();
    atomicAdd(&table[stage * NUM_FIELDS + 0], end - start);
    atomicAdd(&table[stage * NUM_FIELDS + 1], 1ull);
    atomicMin(&table[stage * NUM_FIELDS + 2], start);
    atomicMax(&table[stage * NUM_FIELDS + 3], end);
}

static __host__ std::vector<int64_t> read() {
    std::vector<unsigned long long> host(NUM_STAGES * NUM_FIELDS);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(host.data(), table, sizeof(unsigned long long) * host.size());
    return std::vector<int64_t>(host.begin(), host.end());
}

static __host__ void reset() {
    std::vector<unsigned long long> host(NUM_STAGES * NUM_FIELDS, 0ull);
    for (int s = 0; s < NUM_STAGES; ++s)
        host[s * NUM_FIELDS + 2] = ~0ull;
    cudaDeviceSynchronize();
    cudaMemcpyToSymbol(table, host.data(), sizeof(unsigned long long) * host.size());
}

#define MOK_PROFILE_START(name) const unsigned long long name = ::mok_profile::now()
#define MOK_PROFILE_END(name, stage) ::mok_profile::record((stage), (name))
#else
#define MOK_PROFILE_START(name) do {} while (0)
#define MOK_PROFILE_END(name, stage) do {} while (0)
#endif

} // namespace mok_profile
