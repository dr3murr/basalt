// include/basalt/core/swap_staging.hpp - pinned host slots for expert swap copies whose RAM side is pageable.
//
// A copy between a GPU and pageable RAM holds the thread that issues it until the copy has run, and the driver keeps
// other threads' CUDA calls waiting meanwhile (another card's launches too).  Where the driver does not page-lock the
// whole RAM copy of the experts (WDDM caps page-locked memory near half the RAM, see platform::pin_budget), a swap
// with a pageable RAM place goes through one of these slots instead: its GPU side is a DMA the call only queues, its
// RAM side a memcpy on the thread that owns the swap.  Each slot remembers the last copy queued from or into it;
// `wait` waits for that copy before the host reads or writes the slot again.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

class SwapStaging {
public:
    SwapStaging() = default;
    ~SwapStaging();
    SwapStaging(const SwapStaging&) = delete;
    SwapStaging& operator=(const SwapStaging&) = delete;

    /// `slots` slots of `slot_bytes` each, page-locked for the current device.  Replaces the slots held, so none may
    /// be in use.
    bool reserve(int slots, uint64_t slot_bytes, std::string& err);
    int slots() const { return slots_; }
    /// Slot `i`, 0 <= i < slots().
    uint8_t* slot(int i) const { return base_ + (size_t) i * (size_t) slot_bytes_; }
    /// Waits for the copy queued last from or into slot `i` (returns at once when there was none).
    bool wait(int i, std::string& err) const;
    /// A copy from or into slot `i` was just queued on `stream`.  False when it cannot be marked; the stream has then
    /// been waited for, so the slot is free either way.
    bool queued(int i, cudaStream_t stream, std::string& err);

private:
    void release();
    uint8_t* base_ = nullptr;
    int slots_ = 0;
    uint64_t slot_bytes_ = 0;
    std::vector<cudaEvent_t> done_;   ///< per slot: recorded after the last copy queued from or into it
};

}  // namespace basalt::core
