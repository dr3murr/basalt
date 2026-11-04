// src/kernels/pascal/peer_experts_sm60.cu - the expert tier's kernels for a compute capability 6.0 card (P100).
//
// Built ON ITS OWN by a CUDA 12 nvcc into an sm_60 cubin (CMake option BASALT_PASCAL_NVCC) that the engine embeds
// and loads through the driver API; nothing here is compiled into the sm_120 build.  Arguments:
// include/basalt/kernels/pascal_peer.hpp.
//
// sm_60 has no __dp4a, but full-rate half2 FMA, so the dot products run in fp16 (the GP100 path of our llama.cpp
// tree): activations go to fp16 once, each 32-block scaled by a power of two so that max|y| < 2^15, with the exact
// fp32 block sum kept for the offset terms; weights unpack with byte permutes straight into subnormal half bit
// patterns (q * 2^-24, exact); products accumulate in fp16 over 8-16 terms and are flushed to fp32 with the block
// scale.  The activations keep ~11 bits where the q8_1 path keeps 8, so rows differ from CUDA0's in the last bits.
#include <cuda_fp16.h>
#include <cstdint>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "basalt/ggml/ggml_common.h"

#include "basalt/kernels/pascal_peer.hpp"

using basalt::kernels::pascal::LayerArgs;
using basalt::kernels::pascal::kWarps;

namespace {

constexpr int kWarp = 32;

// Raw PRMT, default mode (__byte_perm masks the selector's sign-replicate bit; the IQ4 lookup never needs it).
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__ uint32_t u32_at(const uint16_t* p) { return (uint32_t) p[0] | ((uint32_t) p[1] << 16); }

// sign byte -> XOR masks for the half2 pairs (w0,w1), (w2,w3), (w4,w5), (w6,w7)
__device__ __forceinline__ uint4 sign_masks(uint32_t s) {
    return make_uint4(((s & 1u) << 15) | ((s & 2u) << 30), ((s & 4u) << 13) | ((s & 8u) << 28),
                      ((s & 16u) << 11) | ((s & 32u) << 26), ((s & 64u) << 9) | ((s & 128u) << 24));
}

constexpr int kNE = basalt::kernels::pascal::kGroupEntries;   // entries one decoded unit is dotted with

// output rows per work item of the gate/up phase (each of gate and up) and of the down phase
#ifndef BP_RG
#define BP_RG 2
#endif
#ifndef BP_RD
#define BP_RD 4
#endif
// blocks per SM the register budget is fitted to (__launch_bounds__' second argument)
#ifndef BP_MINB
#define BP_MINB 2
#endif

template <int NE> using Cols = const half2* [NE];   // each entry's 32 activations of the unit, as 16 half2

template <int NE>
__device__ __forceinline__ void fma_all(uint32_t w, const Cols<NE>& Y, int i, half2 (&acc)[NE]) {
#pragma unroll
    for (int q = 0; q < NE; q++) acc[q] = __hfma2(*(const half2*) &w, Y[q][i], acc[q]);
}

// 8 weights: two 4-byte codebook entries a (w0..3) and b (w4..7), sign masks m; activations Y[.][i .. i+4)
template <int NE>
__device__ __forceinline__ void grid8(uint32_t a, uint32_t b, uint4 m, const Cols<NE>& Y, int i, half2 (&acc)[NE]) {
    fma_all(__byte_perm(a, 0, 0x4140) ^ m.x, Y, i + 0, acc);
    fma_all(__byte_perm(a, 0, 0x4342) ^ m.y, Y, i + 1, acc);
    fma_all(__byte_perm(b, 0, 0x4140) ^ m.z, Y, i + 2, acc);
    fma_all(__byte_perm(b, 0, 0x4342) ^ m.w, Y, i + 3, acc);
}

// 32 IQ4 codes (16 bytes at w: code k low nibble, k+16 high) dotted with Y; the table is looked up +128
template <int NE>
__device__ __forceinline__ void iq4_dot(const uint16_t* w, const Cols<NE>& Y, float (&s)[NE]) {
    constexpr uint32_t T0 = 0x3F2D1801u, T1 = 0x766A5D4Fu, T2 = 0xA6998D81u, T3 = 0xF1D9C5B5u;
    half2 acc[NE];
#pragma unroll
    for (int q = 0; q < NE; q++) acc[q] = __float2half2_rn(0.0f);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint32_t q = u32_at(w + 2 * k);
        const uint32_t qx = q ^ 0x88888888u;
        const uint32_t S = ((q >> 1) & 0x44444444u) | 0x32103210u;
        const uint32_t v0 = prmt(prmt(T0, T1, q), prmt(T2, T3, qx), S);
        const uint32_t v1 = prmt(prmt(T0, T1, q >> 16), prmt(T2, T3, qx >> 16), S >> 16);
        fma_all(__byte_perm(v0, 0, 0x4240), Y, 2 * k + 0, acc);
        fma_all(__byte_perm(v1, 0, 0x4240), Y, 2 * k + 1, acc);
        fma_all(__byte_perm(v0, 0, 0x4341), Y, 8 + 2 * k + 0, acc);
        fma_all(__byte_perm(v1, 0, 0x4341), Y, 8 + 2 * k + 1, acc);
    }
#pragma unroll
    for (int q = 0; q < NE; q++) s[q] = __low2float(acc[q]) + __high2float(acc[q]);
}

// GGML type ids
constexpr int kIQ3_XXS = 18, kIQ4_NL = 20, kIQ3_S = 21, kIQ2_S = 22, kIQ4_XS = 23, kNVFP4 = 40, kQ2_0 = 42;

// NVFP4's UE4M3 sub-block scale as ggml decodes it (half the value, kvalues_fp4 being doubled; 0 and 0x7F as 0, bit 7
// ignored): the same float as the sm_120 kernels' nvfp4_scale_dev.
__device__ __forceinline__ float nvfp4_scale(uint32_t x) {
    const uint32_t v = x & 0x7F;
    if (v == 0 || v == 0x7F) return 0.0f;
    const uint32_t e = v >> 3, m = v & 7;
    if (e == 0) return (float) m * (1.0f / 1024.0f);
    return __uint_as_float(((e + 119) << 23) | (m << 20));
}

// E2M1 code c (bit 3 the sign, bits 0-2 the magnitude) as fp16 bits (c & 7) << 9 | (c & 8) << 12 is exactly the doubled
// E2M1 value (0,1,2,3,4,6,8,12) times 2^-15: e = 0 gives the subnormal m * 2^-15, e >= 1 the normal (2 + m) * 2^(e-16).
// p holds two code bytes, one per 16-bit half (bytes 0 and 2); lo/hi place their low or high nibble.
__device__ __forceinline__ uint32_t fp4_lo(uint32_t p) { return ((p << 9) & 0x0E000E00u) | ((p << 12) & 0x80008000u); }
__device__ __forceinline__ uint32_t fp4_hi(uint32_t p) { return ((p << 5) & 0x0E000E00u) | ((p << 8) & 0x80008000u); }

template <int TY> struct Fmt;
// UPB: 32-weight units per block; GW: codebook words in shared memory; NS: sign-mask entries
template <> struct Fmt<kIQ2_S>   { typedef block_iq2_s block;   static constexpr int UPB = 8, GW = 2048, NS = 256; };
template <> struct Fmt<kIQ3_XXS> { typedef block_iq3_xxs block; static constexpr int UPB = 8, GW = 256, NS = 128; };
template <> struct Fmt<kIQ3_S>   { typedef block_iq3_s block;   static constexpr int UPB = 8, GW = 512, NS = 256; };
template <> struct Fmt<kIQ4_XS>  { typedef block_iq4_xs block;  static constexpr int UPB = 8, GW = 0, NS = 0; };
template <> struct Fmt<kIQ4_NL>  { typedef block_iq4_nl block;  static constexpr int UPB = 1, GW = 0, NS = 0; };
template <> struct Fmt<kQ2_0>    { typedef block_q2_0 block;    static constexpr int UPB = 2, GW = 0, NS = 0; };
template <> struct Fmt<kNVFP4>   { typedef block_nvfp4 block;   static constexpr int UPB = 2, GW = 0, NS = 0; };

// One unit (32 weights: unit j of the block at bb, in shared memory), unpacked once and dotted with each entry's
// matching 32-block of fp16 activations Y[q]; sc[q] = {rescale, fp32 sum of the block's x}.  Adds each entry's
// fp32 contribution to out[q].
template <int TY, int NE>
__device__ __forceinline__ void unit_dot(const uint8_t* bb, int j, const Cols<NE>& Y, const float2 (&sc)[NE],
                                         const uint32_t* grid, const uint4* sgn, float (&out)[NE]) {
    const uint16_t* b16 = (const uint16_t*) bb;
    const float d = __half2float(__ushort_as_half(b16[0]));
    half2 acc[NE];
#pragma unroll
    for (int q = 0; q < NE; q++) acc[q] = __float2half2_rn(0.0f);
    if constexpr (TY == kIQ3_XXS) {
        // qs[0..63]: one grid index per 4 weights; qs[64..95]: per unit, 4 x 7-bit sign indices + 4-bit scale
        const uint32_t q0 = u32_at(b16 + 1 + 4 * j);
        const uint32_t q1 = u32_at(b16 + 3 + 4 * j);
        const uint32_t aux = u32_at(b16 + 33 + 2 * j);
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const uint32_t q = g < 2 ? q0 : q1;
            const int s = 16 * (g % 2);
            grid8(grid[(q >> s) & 0xFF], grid[(q >> (s + 8)) & 0xFF], sgn[(aux >> (7 * g)) & 127], Y, 4 * g, acc);
        }
        const float f = d * (0.5f + (float) (aux >> 28)) * 0.5f;
#pragma unroll
        for (int q = 0; q < NE; q++) out[q] += f * (__low2float(acc[q]) + __high2float(acc[q])) * sc[q].x;
    } else if constexpr (TY == kIQ3_S) {
        // qs[0..63] at 2: low 8 bits of the index per 4 weights; qh[8] at 66; signs[32] at 74; scales[4] at 106
        const uint32_t q0 = u32_at(b16 + 1 + 4 * j);
        const uint32_t q1 = u32_at(b16 + 3 + 4 * j);
        const uint32_t qh = bb[66 + j];
        const uint32_t sg = u32_at(b16 + 37 + 2 * j);
        const int ls = (bb[106 + j / 2] >> (4 * (j & 1))) & 0x0F;
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const uint32_t q = g < 2 ? q0 : q1;
            const int s = 16 * (g % 2);
            const uint32_t ia = ((q >> s) & 0xFF) | ((qh << (8 - 2 * g)) & 0x100);
            const uint32_t ib = ((q >> (s + 8)) & 0xFF) | ((qh << (7 - 2 * g)) & 0x100);
            grid8(grid[ia], grid[ib], sgn[(sg >> (8 * g)) & 0xFF], Y, 4 * g, acc);
        }
        const float f = d * (float) (1 + 2 * ls);
#pragma unroll
        for (int q = 0; q < NE; q++) out[q] += f * (__low2float(acc[q]) + __high2float(acc[q])) * sc[q].x;
    } else if constexpr (TY == kIQ2_S) {
        // qs[0..31] at 2: low 8 bits of the index per 8 weights; qs[32..63] at 34: sign bytes; qh[8] at 66; scales at 74
        const uint32_t qi = u32_at(b16 + 1 + 2 * j);
        const uint32_t sg = u32_at(b16 + 17 + 2 * j);
        const uint32_t qh = bb[66 + j];
        const uint32_t ls = bb[74 + j];
        half2 acc1[NE];
#pragma unroll
        for (int q = 0; q < NE; q++) acc1[q] = __float2half2_rn(0.0f);
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const uint32_t idx = ((qi >> (8 * g)) & 0xFF) | ((qh << (8 - 2 * g)) & 0x300);
            const uint2 gr = *(const uint2*) (grid + 2 * idx);
            if (g < 2) grid8(gr.x, gr.y, sgn[(sg >> (8 * g)) & 0xFF], Y, 4 * g, acc);
            else       grid8(gr.x, gr.y, sgn[(sg >> (8 * g)) & 0xFF], Y, 4 * g, acc1);
        }
        const float f0 = d * 0.25f * (0.5f + (float) (ls & 0x0F)), f1 = d * 0.25f * (0.5f + (float) (ls >> 4));
#pragma unroll
        for (int q = 0; q < NE; q++)
            out[q] += (f0 * (__low2float(acc[q]) + __high2float(acc[q])) +
                       f1 * (__low2float(acc1[q]) + __high2float(acc1[q]))) * sc[q].x;
    } else if constexpr (TY == kIQ4_XS || TY == kIQ4_NL) {
        float s[NE];
        float dl = d;
        if constexpr (TY == kIQ4_XS) {
            // d, scales_h (u16) at 2, scales_l[4] at 4, qs[128] at 8: unit j's codes are qs[16j .. 16j+16)
            const uint32_t sh = b16[1];
            const int ls = ((bb[4 + j / 2] >> (4 * (j & 1))) & 0x0F) | (((sh >> (2 * j)) & 3) << 4);
            dl = d * (float) (ls - 32);
            iq4_dot(b16 + 4 + 8 * j, Y, s);
        } else {
            iq4_dot(b16 + 1, Y, s);
        }
#pragma unroll
        for (int q = 0; q < NE; q++) out[q] += dl * fmaf(s[q], sc[q].x, -128.0f * sc[q].y);
    } else if constexpr (TY == kNVFP4) {
        // d[4] (UE4M3) at 0, qs[32] at 4; unit j = sub-blocks 2j and 2j+1, sub-block s's codes at qs[8s .. 8s+8):
        // byte k holds value k (low nibble) and value k + 8 (high nibble).  One fp16 sum per sub-block (own scale).
        half2 acc1[NE];
#pragma unroll
        for (int q = 0; q < NE; q++) acc1[q] = __float2half2_rn(0.0f);
#pragma unroll
        for (int h = 0; h < 2; h++) {
#pragma unroll
            for (int w = 0; w < 2; w++) {
                const uint32_t qw = u32_at(b16 + 2 + 8 * j + 4 * h + 2 * w);
                const int i = 8 * h + 2 * w;   // half2 index of value 4w of the sub-block
                const uint32_t p01 = __byte_perm(qw, 0, 0x4140), p23 = __byte_perm(qw, 0, 0x4342);
                const uint32_t lo01 = fp4_lo(p01), lo23 = fp4_lo(p23), hi01 = fp4_hi(p01), hi23 = fp4_hi(p23);
                if (h == 0) {
                    fma_all(lo01, Y, i, acc); fma_all(lo23, Y, i + 1, acc);
                    fma_all(hi01, Y, i + 4, acc); fma_all(hi23, Y, i + 5, acc);
                } else {
                    fma_all(lo01, Y, i, acc1); fma_all(lo23, Y, i + 1, acc1);
                    fma_all(hi01, Y, i + 4, acc1); fma_all(hi23, Y, i + 5, acc1);
                }
            }
        }
        // the weights are 2^-15 times the doubled value; sc[].x undoes the 2^-24 of the other formats' encoding
        const float f0 = nvfp4_scale(bb[2 * j]) * 0x1p-9f, f1 = nvfp4_scale(bb[2 * j + 1]) * 0x1p-9f;
#pragma unroll
        for (int q = 0; q < NE; q++)
            out[q] += (f0 * (__low2float(acc[q]) + __high2float(acc[q])) +
                       f1 * (__low2float(acc1[q]) + __high2float(acc1[q]))) * sc[q].x;
    } else {   // Q2_0: 64 weights per block, code c (2 bits, weight i at bit 2i of qs) means c - 1; unit j = weights 32j..
#pragma unroll
        for (int h = 0; h < 2; h++) {   // 16 weights per 32-bit word
            const uint32_t w = u32_at(b16 + 1 + 4 * j + 2 * h);
#pragma unroll
            for (int k = 0; k < 4; k++) {
                // byte b of x holds weight 4b+k: halves (k, 4+k) and (8+k, 12+k)
                const uint32_t x = (w >> (2 * k)) & 0x03030303u;
                const uint32_t pa = __byte_perm(x, 0, 0x4140);
                const uint32_t pb = __byte_perm(x, 0, 0x4342);
#pragma unroll
                for (int q = 0; q < NE; q++) {
                    const uint32_t* Yw = (const uint32_t*) (Y[q] + 8 * h);
                    const uint32_t ya = __byte_perm(Yw[k / 2], Yw[2 + k / 2], (k & 1) ? 0x7632 : 0x5410);
                    const uint32_t yb = __byte_perm(Yw[4 + k / 2], Yw[6 + k / 2], (k & 1) ? 0x7632 : 0x5410);
                    acc[q] = __hfma2(*(const half2*) &pa, *(const half2*) &ya, acc[q]);
                    acc[q] = __hfma2(*(const half2*) &pb, *(const half2*) &yb, acc[q]);
                }
            }
        }
#pragma unroll
        for (int q = 0; q < NE; q++) out[q] += d * fmaf(__low2float(acc[q]) + __high2float(acc[q]), sc[q].x, -sc[q].y);
    }
}

// Every block of the grid arrives; returns once all have (the grid is co-resident: launched with at most the
// blocks that fit at once).  The counter is never reset: each launch's targets start at its `bar_base`.
__device__ __forceinline__ void grid_barrier(unsigned long long* bar, unsigned long long target) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(bar, 1ull);
        while (*(volatile unsigned long long*) bar < target) {
        }
        __threadfence();
    }
    __syncthreads();
}

// x (fp32, 32-blocks) -> y (fp16) with y = x * 2^k per 32-block, max|y| < 2^15; ys[b] = {2^(24-k), sum of the
// block's x} (2^24 undoes the subnormal weight encoding).  One warp per 32-block, over the whole grid.  Plain loads:
// `x` may have been written earlier in this launch.
__device__ __forceinline__ void prep_rows(const float* x, half* y, float2* ys, int nblocks) {
    const int lane = threadIdx.x % kWarp;
    for (int b = blockIdx.x * kWarps + threadIdx.x / kWarp; b < nblocks; b += gridDim.x * kWarps) {
        const float v = x[(int64_t) b * 32 + lane];
        float m = fabsf(v), s = v;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
            s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
        }
        int e = 0;
        if (m > 0.0f) frexpf(m, &e);   // m in [2^(e-1), 2^e)
        const int k = 15 - e;
        y[(int64_t) b * 32 + lane] = __float2half_rn(ldexpf(v, k));
        if (lane == 0) ys[b] = make_float2(ldexpf(1.0f, 24 - k), s);
    }
}

template <int TY> struct Shape {
    static constexpr int BS = sizeof(typename Fmt<TY>::block);
    static constexpr int QK = 32 * Fmt<TY>::UPB;
    static constexpr int CB = 32 / Fmt<TY>::UPB;          // blocks per chunk: one 32-weight unit per lane
    static constexpr int CW = (CB * BS + 3) / 4;           // 32-bit words per chunk and row
    static constexpr int NL = (CW + kWarp - 1) / kWarp;    // of them per lane
    static_assert((CB * BS) % 4 == 0, "a chunk must be whole words");
};

// (kNE: the entries of a group, at most kGroupEntries; above)

// A work item's expert: grp[g] = {blob address (lo, hi), first entry | entries << 16, token of each entry (a byte)}.
struct Item {
    const char* blob;
    int row0, e0, ne;
    uint32_t toks;
};

// One grouped matmul phase.  Work items are (group, R output rows), spread over every warp of the grid; a warp walks
// them step by step, a step being one chunk of its rows (one 32-weight unit per lane) dotted with each of the group's
// entries, so the expert's weights are read once for them.  The next step's weights and activations are loaded
// before the current step is decoded, across items too, and the next item's group is read one item ahead.
// GLU: rows [0, R) of gate and the same R rows of up (at off1), stored as silu(gate) * up.
// Activations: column `token` (by_tok) or `entry` of y/ys.  Output: row `entry` of dst (n_out floats).
template <int TY, int R, bool GLU>
__device__ __forceinline__ void mm_phase(const LayerArgs& a, const uint4* grp, uint32_t* stage, const uint32_t* grid,
                                         const uint4* sgn, int n_in, int n_out, long long row_bytes, long long off0,
                                         long long off1, const half* y, const float2* ys, bool by_tok, float* dst) {
    typedef Shape<TY> S;
    constexpr int RT = GLU ? 2 * R : R;
    constexpr int UPB = Fmt<TY>::UPB;
    const int lane = threadIdx.x % kWarp;
    const int nblk = n_in / S::QK;
    const int nb32 = n_in / 32;
    const int nrg = (n_out + R - 1) / R;
    const int nchunk = (nblk + S::CB - 1) / S::CB;
    const int items = a.groups * nrg;
    const int stride = gridDim.x * kWarps;
    const auto raw = [&](int it) { return it < items ? grp[it / nrg] : make_uint4(0, 0, 0, 0); };
    const auto item = [&](int it, uint4 gv) {
        Item m;
        m.blob = (const char*) (((unsigned long long) gv.y << 32) | gv.x);
        m.row0 = (it % nrg) * R;
        m.e0 = (int) (gv.z & 0xffffu);
        m.ne = (int) (gv.z >> 16);
        m.toks = gv.w;
        return m;
    };
    // a step's weights: its rows' chunk, one 32-bit word per lane and row at a time (coalesced)
    const auto load = [&](const Item& m, int c, uint32_t (&v)[RT][S::NL]) {
        const int b0 = c * S::CB;
        const int nw = (min(nblk - b0, S::CB) * S::BS + 3) / 4;
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const uint32_t* src = (const uint32_t*) (m.blob + ((GLU && r >= R) ? off1 : off0) +
                                                     (int64_t) min(m.row0 + r % R, n_out - 1) * row_bytes) +
                                  b0 * S::BS / 4;
#pragma unroll
            for (int k = 0; k < S::NL; k++) {
                const int i = lane + kWarp * k;
                v[r][k] = i < nw ? __ldg(src + i) : 0u;
            }
        }
    };

    int it = blockIdx.x * kWarps + threadIdx.x / kWarp;
    if (it >= items) return;
    Item m = item(it, raw(it));
    uint4 next_raw = raw(it + stride);
    int c = 0;
    uint32_t v[RT][S::NL];
    load(m, 0, v);
    float sum[kNE][RT];
#pragma unroll
    for (int q = 0; q < kNE; q++)
#pragma unroll
        for (int r = 0; r < RT; r++) sum[q][r] = 0.0f;
    while (true) {
        int c_n = c + 1, it_n = it;
        Item mn = m;
        if (c_n == nchunk) {
            c_n = 0;
            it_n = it + stride;
            mn = item(it_n, next_raw);
            next_raw = raw(it_n + stride);   // used one item later: its latency hides behind this item's steps
        }
        uint32_t vn[RT][S::NL];
        if (it_n < items) load(mn, c_n, vn);

        __syncwarp();   // the previous step's decode is done with the stage
#pragma unroll
        for (int r = 0; r < RT; r++)
#pragma unroll
            for (int k = 0; k < S::NL; k++) {
                const int i = lane + kWarp * k;
                if (i < S::CW) stage[r * S::CW + i] = v[r][k];
            }
        __syncwarp();
        const int b0 = c * S::CB;
        if (lane < min(nblk - b0, S::CB) * UPB) {
            const int bi = lane / UPB, j = lane % UPB;
            // the entries' activations: written before this phase's grid barrier and read by every warp of the
            // block, so through the non-coherent L1 path (nothing of them is cached from before the barrier)
            const int b = b0 * UPB + lane;
            // decode once per unit, dotted with only the group's entries (one entry: half the FMAs and loads)
            const auto decode = [&](auto ne_c) {
                constexpr int NE = decltype(ne_c)::value;
                uint4 av[NE][4];
                float2 as[NE];
                Cols<NE> Y;
#pragma unroll
                for (int q = 0; q < NE; q++) {
                    const int col = by_tok ? (int) ((m.toks >> (8 * q)) & 0xffu) : m.e0 + q;
                    const uint4* yp = (const uint4*) (y + (int64_t) col * n_in + 32 * b);
#pragma unroll
                    for (int k = 0; k < 4; k++) av[q][k] = __ldg(yp + k);
                    as[q] = __ldg(ys + (int64_t) col * nb32 + b);
                    Y[q] = (const half2*) av[q];
                }
#pragma unroll
                for (int r = 0; r < RT; r++) {
                    float o[NE];
#pragma unroll
                    for (int q = 0; q < NE; q++) o[q] = 0.0f;
                    unit_dot<TY>((const uint8_t*) (stage + r * S::CW) + bi * S::BS, j, Y, as, grid, sgn, o);
#pragma unroll
                    for (int q = 0; q < NE; q++) sum[q][r] += o[q];
                }
            };
            static_assert(kNE == 2, "the decode dispatch handles groups of 1 or 2 entries");
            if (m.ne == 1) decode(std::integral_constant<int, 1>{});
            else           decode(std::integral_constant<int, 2>{});
        }
        if (c == nchunk - 1) {
#pragma unroll
            for (int q = 0; q < kNE; q++) {
#pragma unroll
                for (int r = 0; r < RT; r++) {
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) sum[q][r] += __shfl_xor_sync(0xFFFFFFFFu, sum[q][r], o);
                }
                if (lane == 0 && q < m.ne) {
                    // NVFP4 blobs end in FP32 {s_gate, s_up, s_down, 0}, each on its matrix's output (gate before SiLU)
                    float sa = 1.0f, sb = 1.0f;
                    if constexpr (TY == kNVFP4) {
                        const float* t = (const float*) (m.blob + a.down_off + (int64_t) a.n_embd * a.d_row);
                        sa = __ldg(t + (GLU ? 0 : 2));
                        sb = __ldg(t + 1);
                    }
#pragma unroll
                    for (int r = 0; r < R; r++)
                        if (m.row0 + r < n_out) {
                            float o = sum[q][r] * sa;
                            if (GLU) o = o / (1.0f + __expf(-o)) * (sum[q][R + r] * sb);
                            dst[(int64_t) (m.e0 + q) * n_out + m.row0 + r] = o;
                        }
                }
#pragma unroll
                for (int r = 0; r < RT; r++) sum[q][r] = 0.0f;
            }
        }
        if (it_n >= items) break;
#pragma unroll
        for (int r = 0; r < RT; r++)
#pragma unroll
            for (int k = 0; k < S::NL; k++) v[r][k] = vn[r][k];
        m = mn;
        it = it_n;
        c = c_n;
    }
}

// One layer's peer entries: the tokens' activations to fp16, gate/up + SwiGLU, h to fp16, down, and with host_out
// the rows to the host, phase after phase over the whole (co-resident) grid.
template <int TG, int TD>
__device__ __forceinline__ void layer(const LayerArgs& a) {
    static_assert(Fmt<TD>::GW == 0, "a down type with a codebook needs its table in shared memory too");
    constexpr int GW = Fmt<TG>::GW > 0 ? Fmt<TG>::GW : 1;
    constexpr int NS = Fmt<TG>::NS > 0 ? Fmt<TG>::NS : 1;
    constexpr int SG = 2 * BP_RG * Shape<TG>::CW, SD = BP_RD * Shape<TD>::CW;   // a work item's rows, one chunk each
    constexpr int SW = SG > SD ? SG : SD;
    __shared__ uint32_t stage[kWarps][SW];
    __shared__ __align__(16) uint32_t grid[GW];
    __shared__ uint4 sgn[NS];
    if constexpr (Fmt<TG>::GW > 0) {
        const uint32_t* src = TG == kIQ2_S ? (const uint32_t*) iq2s_grid : TG == kIQ3_XXS ? iq3xxs_grid : iq3s_grid;
        for (int i = threadIdx.x; i < Fmt<TG>::GW; i += blockDim.x) grid[i] = src[i];
        for (int i = threadIdx.x; i < Fmt<TG>::NS; i += blockDim.x) {
            // IQ3_XXS stores 7 sign bits, the 8th makes the count even
            const uint32_t s = TG == kIQ3_XXS ? i | ((__popc(i) & 1) << 7) : i;
            sgn[i] = sign_masks(s);
        }
    }
    // the plan (in mapped host memory: no copy engine in the decode path) into shared memory, read once per block
    __shared__ uint4 sgrp[basalt::kernels::pascal::kMaxEntries];
    __shared__ int32_t srow[basalt::kernels::pascal::kMaxEntries];
    for (int i = threadIdx.x; i < a.groups; i += blockDim.x) sgrp[i] = ((const uint4*) a.grp)[i];
    for (int i = threadIdx.x; i < a.entries; i += blockDim.x) srow[i] = a.out_row[i];
    uint32_t* st = stage[threadIdx.x / kWarp];
    const unsigned long long nb = gridDim.x;
    half* y = (half*) a.y;
    float2* ys = (float2*) a.ys;
    half* hy = (half*) a.hy;
    float2* hys = (float2*) a.hys;
    // stamps (diagnostics, null in the engine): [0, 6) block 0's globaltimer at the start and after each phase;
    // [8 + b] the last phase block b reached (1 + its index), written before the phase starts
    const auto stamp = [&](int i) {
        if (a.stamps != nullptr && threadIdx.x == 0) {
            if (blockIdx.x == 0) {
                unsigned long long t;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                a.stamps[i] = t;
            }
            *(volatile unsigned long long*) (a.stamps + 8 + blockIdx.x) = (unsigned long long) i + 1;
            __threadfence_system();
        }
    };
    stamp(0);
    prep_rows(a.x, y, ys, a.n_tok * a.n_embd / 32);
    grid_barrier(a.bar, a.bar_base + nb);   // also orders the tables' and the plan's fill
    stamp(1);
    mm_phase<TG, BP_RG, true>(a, sgrp, st, grid, sgn, a.n_embd, a.n_ff, a.gu_row, 0, a.up_off, y, ys, true, a.h);
    grid_barrier(a.bar, a.bar_base + 2 * nb);
    stamp(2);
    prep_rows(a.h, hy, hys, a.entries * a.n_ff / 32);
    grid_barrier(a.bar, a.bar_base + 3 * nb);
    stamp(3);
    mm_phase<TD, BP_RD, false>(a, sgrp, st, grid, sgn, a.n_ff, a.n_embd, a.d_row, a.down_off, 0, hy, hys, false, a.out);
    grid_barrier(a.bar, a.bar_base + 4 * nb);
    stamp(4);
    if (a.host_out != nullptr) {   // compact row e to row out_row[e], coalesced float4 stores over the link
        const int n4 = a.n_embd / 4;
        const int total = a.entries * n4;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
            const int r = i / n4, c = i - r * n4;
            ((float4*) a.host_out)[(int64_t) srow[r] * n4 + c] = ((const float4*) a.out)[i];
        }
    }
    stamp(5);
}

}  // namespace

#define BP_LAYER(G, D) \
    extern "C" __global__ void __launch_bounds__(kWarps * 32, BP_MINB) bp_layer_##G##_##D(LayerArgs a) { layer<G, D>(a); }
BP_LAYER(18, 20) BP_LAYER(21, 20) BP_LAYER(22, 20) BP_LAYER(23, 20)
BP_LAYER(18, 42) BP_LAYER(21, 42) BP_LAYER(22, 42) BP_LAYER(23, 42)
BP_LAYER(40, 40)
