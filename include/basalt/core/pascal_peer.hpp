// include/basalt/core/pascal_peer.hpp - the expert tier on a compute capability 6.x card (Tesla P100).
//
// The engine is built for sm_120 only, so none of its kernels run on a Pascal card.  A Pascal expert tier computes
// its rows with kernels of its own (src/kernels/pascal/peer_experts_sm60.cu), built by a CUDA 12 toolchain into an
// sm_60 cubin that is embedded in the binary (CMake option BASALT_PASCAL_NVCC) and loaded through the driver API on
// that card.  Everything else the tier does (its slots, fills, swaps, the host rows) is the runtime API as before.
// Such a tier serves the decode windows' pool path only: the device-planned MoE tail, the prompt chunks' peer share
// and the prompt-time split stay on the main card with a Pascal tier.
#pragma once

#include "basalt/kernels/pascal_peer.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace basalt::core {

/// The binary carries the Pascal tier's kernels (built with BASALT_PASCAL_NVCC).
bool pascal_peer_built();
/// Whether CUDA device `device` is a Pascal card (compute capability 6.x).
bool is_pascal_device(int device);

/// The Pascal kernels, loaded on one card.  Every call runs on the caller's current device, which must be the card
/// `load` was given.
class PascalKernels {
public:
    PascalKernels() = default;
    ~PascalKernels();
    PascalKernels(const PascalKernels&) = delete;
    PascalKernels& operator=(const PascalKernels&) = delete;

    bool load(int device, std::string& err);
    /// Whether the cubin has a layer kernel for this gate/up and down type pair.
    bool supports(int gu_type, int d_type) const;
    const void* tmp_bar() const { return bar_; }   // TMP_PASCAL_DEBUG
    /// One layer's entries (kernels::pascal::LayerArgs; `bar`, `bar_base` and `stamps` are filled in here), queued on
    /// `s` with as many blocks as are resident at once.
    bool layer(int gu_type, int d_type, basalt::kernels::pascal::LayerArgs a, cudaStream_t s, std::string& err);

private:
    static constexpr int kTypes = 64;
    void* ctx_ = nullptr;     ///< the card's primary context, made current for each launch
    void* mod_ = nullptr;
    unsigned long long* bar_ = nullptr;
    unsigned long long bar_base_ = 0;
    void* fn_[kTypes][kTypes] = {};
    int blocks_[kTypes][kTypes] = {};
};

}  // namespace basalt::core
