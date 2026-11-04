// include/basalt/kernels/pascal_peer.hpp - the kernel arguments of the Pascal (sm_60) expert tier.
//
// The Pascal kernels (src/kernels/pascal/peer_experts_sm60.cu) are built by a CUDA 12 toolchain into a cubin that
// the engine embeds and loads with the driver API on a compute capability 6.x card; the engine itself is built for
// sm_120 by CUDA 13, which compiles nothing older than sm_75.  This header is the one thing both builds share, so
// it holds plain data only.
#pragma once

#include <cstdint>

namespace basalt::kernels::pascal {

inline constexpr int kGroupEntries = 2;   ///< the most entries of one group (split a larger one into groups)
inline constexpr int kMaxEntries = 128;   ///< the most entries (and so groups) of one launch

/// One group of a layer's peer entries: an expert blob ([gate rows | up rows | down rows], raw blocks) and up to
/// kGroupEntries consecutive entries routed to it.
struct Group {
    unsigned long long blob;
    uint32_t first_and_count;   ///< first entry | entries << 16
    uint32_t tokens;            ///< the token of each entry, a byte each
};
static_assert(sizeof(Group) == 16, "a group is one 16-byte load");

/// One layer's peer entries in one launch of `bp_layer_G_D` (G: the gate/up GGML type, D: the down type).  Entry
/// e's result row (down of silu(gate . x) * (up . x), n_embd floats) goes to row e of `out` and, with `host_out`,
/// to row out_row[e] there.  The kernel runs its phases with grid barriers, so it is launched with at most the
/// blocks that are resident at once.
struct LayerArgs {
    const void* grp;           ///< `groups` Group records (16-byte aligned; device or mapped host memory)
    const int32_t* out_row;    ///< device or mapped host memory
    const float* x;            ///< n_tok x n_embd (device or mapped host memory)
    void* y;                   ///< scratch: n_tok x n_embd halves
    void* ys;                  ///< scratch: n_tok x n_embd/32 float2
    float* h;                  ///< scratch: entries x n_ff
    void* hy;                  ///< scratch: entries x n_ff halves
    void* hys;                 ///< scratch: entries x n_ff/32 float2
    float* out;                ///< entries x n_embd
    float* host_out;           ///< mapped host rows, or null
    unsigned long long* bar;   ///< the grid barrier's counter (never reset: `bar_base` moves on)
    unsigned long long bar_base;
    long long gu_row, d_row;   ///< bytes per gate/up row and per down row
    long long up_off, down_off;
    int32_t groups, entries, n_tok;
    int32_t n_embd, n_ff;
    unsigned long long* stamps;   ///< null, or 6 globaltimer stamps: start and the end of each phase (block 0)
    /// mapped host rows as fp16, compact (entry e at row e), each scaled by a power of two so its largest value is in
    /// [2^14, 2^15); row e's values are its halves times `half_scale[e]` (mapped host memory).  Or null.
    void* host_half;
    float* half_scale;
};

inline constexpr int kWarps = 8;      ///< warps per block
inline constexpr int kBarriers = 4;   ///< grid barriers per launch (bar_base advances by kBarriers * blocks)

}  // namespace basalt::kernels::pascal
