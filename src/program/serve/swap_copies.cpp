// src/program/serve/swap_copies.cpp - the adaptive rounds' copies that touch pageable RAM, on xq_thr through the
// pinned bounce slots (see basalt/core/swap_staging.hpp).
#include "state.hpp"

#include <algorithm>
#include <cstring>

namespace basalt::program::serve {

bool Runtime::xq_join_copies() {
    if (!xq_job) return true;
    {
        const GapSpan span{kGapCopierWait};
        xq_thr.join();
    }
    xq_job = false;
    if (xq_copy_err.empty()) return true;
    basalt::log::error("basalt serve", "%s", xq_copy_err.c_str());
    xq_copy_err.clear();
    return false;
}

// `job` runs on xq_thr, on the caller's device; xq_join_copies waits for it
void Runtime::xq_hand_off(std::function<void()> job) {
    int dev = 0;
    cudaGetDevice(&dev);
    xq_thr.run([job = std::move(job), dev] {
        const GapSpan span{kGapCopier};
        cudaSetDevice(dev);
        job();
    });
    xq_job = true;
}

// The round's evictions into pageable places: each lands by DMA in a bounce slot and is copied from there into its
// place, at most a slot count of them in flight.  adapt_ev follows the last DMA.
void Runtime::xq_queue_evictions() {
    const auto& lay = basalt::kernels::cpu::expert_layout();
    std::vector<size_t> order;   // the round's swaps whose eviction goes through a slot
    for (size_t i = 0; i < xq.size(); ++i)
        if (xq[i].evict_to != nullptr) order.push_back(i);
    const size_t n_slots = (size_t) xq_bounce.slots();
    size_t queued = 0;
    for (size_t landed = 0; landed < order.size(); ++landed) {
        // slot q % n_slots last held eviction q - n_slots, which is in its place already
        for (; queued < order.size() && queued < landed + n_slots; ++queued) {
            const XSwap& x = xq[order[queued]];
            const int s = (int) (queued % n_slots);
            if (cudaMemcpyAsync(xq_bounce.slot(s), xcache.device_slot(x.slot), (size_t) lay.blob_bytes(x.lout),
                                cudaMemcpyDeviceToHost, adapt_stream) != cudaSuccess) {
                xq_copy_err = "an eviction copy into a bounce slot failed (layer " + std::to_string(x.lout) + ", slot " +
                              std::to_string(x.slot) + "): " + cudaGetErrorString(cudaGetLastError());
                return;
            }
            if (!xq_bounce.queued(s, adapt_stream, xq_copy_err)) return;
        }
        const XSwap& x = xq[order[landed]];
        const int s = (int) (landed % n_slots);
        if (!xq_bounce.wait(s, xq_copy_err)) return;
        std::memcpy(x.evict_to, xq_bounce.slot(s), (size_t) lay.blob_bytes(x.lout));
    }
    cudaEventRecord(adapt_ev, adapt_stream);
}

// The round's incoming experts into their GPU slots in layer order, each layer's landed word right after its own
// copies (`words`: the round's words, or null).  A pageable source is copied into the next bounce slot in turn, once
// the DMA out of it has run.  A slot that fails leaves the rest of the round copying straight from RAM, so every
// copy and landed word is still queued (the next window waits for the words).  adapt_ev follows the last copy.
void Runtime::xq_queue_incoming(const uint32_t* words) {
    const auto& lay = basalt::kernels::cpu::expert_layout();
    std::vector<uint32_t> ord;
    for (uint32_t i = 0; i < (uint32_t) xq.size(); ++i)
        if (!xq[i].p2p) ord.push_back(i);
    std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return xq[a].layer < xq[b].layer; });
    bool bounce = xq_bounce.slots() > 0;
    int turn = 0;
    std::string serr;
    auto stop_bouncing = [&] {
        basalt::log::warn("basalt serve", "%s; the round's other copies go straight from RAM", serr.c_str());
        bounce = false;
    };
    for (size_t k = 0; k < ord.size(); ++k) {
        const XSwap& x = xq[ord[k]];
        const size_t bytes = (size_t) lay.blob_bytes(x.layer);
        const uint8_t* from = x.from;
        int s = -1;
        if (bounce && from != nullptr && !srcp->pinned(x.layer, x.in)) {
            s = turn++ % xq_bounce.slots();
            if (xq_bounce.wait(s, serr)) {
                std::memcpy(xq_bounce.slot(s), from, bytes);
                from = xq_bounce.slot(s);
            } else {
                stop_bouncing();
                s = -1;
            }
        }
        if (from == nullptr || cudaMemcpyAsync(xcache.device_slot(x.slot), from, bytes, cudaMemcpyHostToDevice,
                                               adapt_stream) != cudaSuccess) {
            basalt::log::warn("basalt serve", "adaptive swap copy failed (layer %d, slot %d): %s", (int) x.layer,
                              (int) x.slot, cudaGetErrorString(cudaGetLastError()));
        } else if (s >= 0 && !xq_bounce.queued(s, adapt_stream, serr)) {
            stop_bouncing();
        }
        if (words != nullptr && (k + 1 == ord.size() || xq[ord[k + 1]].layer != x.layer) &&
            cudaMemcpyAsync(ver.landed_words() + x.layer, words + g.n_layers, 4, cudaMemcpyHostToDevice, adapt_stream) !=
                cudaSuccess)
            basalt::log::warn("basalt serve", "the landed word's copy failed (layer %d): %s", (int) x.layer,
                              cudaGetErrorString(cudaGetLastError()));
    }
    cudaEventRecord(adapt_ev, adapt_stream);
}

}  // namespace basalt::program::serve
