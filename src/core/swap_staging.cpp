// src/core/swap_staging.cpp - see include/basalt/core/swap_staging.hpp.
#include "basalt/core/swap_staging.hpp"

namespace basalt::core {

namespace {
bool fail(std::string& err, const std::string& what, cudaError_t e) {
    (void) cudaGetLastError();
    err = "swap staging: " + what + " (" + cudaGetErrorString(e) + ")";
    return false;
}
}  // namespace

SwapStaging::~SwapStaging() { release(); }

void SwapStaging::release() {
    for (cudaEvent_t e : done_)
        if (e != nullptr) (void) cudaEventDestroy(e);
    done_.clear();
    if (base_ != nullptr) (void) cudaFreeHost(base_);
    base_ = nullptr;
    slots_ = 0;
    slot_bytes_ = 0;
}

bool SwapStaging::reserve(int slots, uint64_t slot_bytes, std::string& err) {
    release();
    if (slots <= 0 || slot_bytes == 0) {
        err = "swap staging: no slots asked for";
        return false;
    }
    const uint64_t bytes = (uint64_t) slots * slot_bytes;
    void* p = nullptr;
    const cudaError_t e = cudaHostAlloc(&p, (size_t) bytes, cudaHostAllocDefault);
    if (e != cudaSuccess) return fail(err, "cannot page-lock " + std::to_string(bytes >> 20) + " MiB", e);
    base_ = (uint8_t*) p;
    done_.assign((size_t) slots, nullptr);
    for (cudaEvent_t& ev : done_) {
        const cudaError_t ee = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (ee != cudaSuccess) {
            ev = nullptr;
            release();
            return fail(err, "cannot create a slot's event", ee);
        }
    }
    slots_ = slots;
    slot_bytes_ = slot_bytes;
    return true;
}

bool SwapStaging::wait(int i, std::string& err) const {
    const cudaError_t e = cudaEventSynchronize(done_[(size_t) i]);
    return e == cudaSuccess || fail(err, "slot " + std::to_string(i) + "'s last copy failed", e);
}

bool SwapStaging::queued(int i, cudaStream_t stream, std::string& err) {
    const cudaError_t e = cudaEventRecord(done_[(size_t) i], stream);
    if (e == cudaSuccess) return true;
    (void) cudaStreamSynchronize(stream);   // unmarked, the slot is free only once its copy has run
    return fail(err, "cannot mark slot " + std::to_string(i) + "'s copy", e);
}

}  // namespace basalt::core
