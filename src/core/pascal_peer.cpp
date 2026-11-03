// src/core/pascal_peer.cpp - see include/basalt/core/pascal_peer.hpp.
#include "basalt/core/pascal_peer.hpp"

#include <cuda.h>   // driver API types only: the entry points come from the runtime, so nothing links libcuda

#include <mutex>

#ifdef BASALT_PASCAL_PEER
extern const unsigned char basalt_pascal_cubin[];
extern const unsigned long long basalt_pascal_cubin_size;
#endif

namespace basalt::core {

namespace {

namespace kp = basalt::kernels::pascal;

struct Driver {
    decltype(&cuModuleLoadData) load = nullptr;
    decltype(&cuModuleUnload) unload = nullptr;
    decltype(&cuModuleGetFunction) get = nullptr;
    decltype(&cuLaunchKernel) launch = nullptr;
    decltype(&cuGetErrorString) error = nullptr;
    decltype(&cuCtxGetCurrent) ctx_get = nullptr;
    decltype(&cuCtxSetCurrent) ctx_set = nullptr;
    decltype(&cuOccupancyMaxActiveBlocksPerMultiprocessor) occupancy = nullptr;
    bool ok = false;
};

const Driver& driver() {
    static Driver d;
    static std::once_flag once;
    std::call_once(once, [] {
        auto sym = [](const char* name, void** fn) {
            cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
            return cudaGetDriverEntryPointByVersion(name, fn, 12000, cudaEnableDefault, &q) == cudaSuccess &&
                   q == cudaDriverEntryPointSuccess && *fn != nullptr;
        };
        d.ok = sym("cuModuleLoadData", (void**) &d.load) && sym("cuModuleUnload", (void**) &d.unload) &&
               sym("cuModuleGetFunction", (void**) &d.get) && sym("cuLaunchKernel", (void**) &d.launch) &&
               sym("cuGetErrorString", (void**) &d.error) && sym("cuCtxGetCurrent", (void**) &d.ctx_get) &&
               sym("cuCtxSetCurrent", (void**) &d.ctx_set) &&
               sym("cuOccupancyMaxActiveBlocksPerMultiprocessor", (void**) &d.occupancy);
        if (!d.ok) cudaGetLastError();
    });
    return d;
}

bool dk(CUresult r, const char* what, std::string& err) {
    if (r == CUDA_SUCCESS) return true;
    const char* s = nullptr;
    if (driver().error != nullptr) driver().error(r, &s);
    err = std::string("pascal tier: ") + what + ": " + (s != nullptr ? s : "driver error " + std::to_string((int) r));
    return false;
}

constexpr int kGuTypes[] = {18, 21, 22, 23};   // the cubin's bp_layer_G_D
constexpr int kDownTypes[] = {20, 42};

}  // namespace

bool pascal_peer_built() {
#ifdef BASALT_PASCAL_PEER
    return true;
#else
    return false;
#endif
}

bool is_pascal_device(int device) {
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return major == 6;
}

PascalKernels::~PascalKernels() {
    if (bar_ != nullptr) cudaFree(bar_);
    if (mod_ != nullptr && driver().unload != nullptr) driver().unload((CUmodule) mod_);
}

bool PascalKernels::load(int device, std::string& err) {
#ifndef BASALT_PASCAL_PEER
    (void) device;
    err = "pascal tier: this build has no Pascal kernels (configure with -DBASALT_PASCAL_NVCC=<a CUDA 12 nvcc>)";
    return false;
#else
    const Driver& d = driver();
    if (!d.ok) { err = "pascal tier: the driver API entry points are not available"; return false; }
    int nsm = 0;
    if (cudaSetDevice(device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess ||
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaMalloc((void**) &bar_, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMemset(bar_, 0, sizeof(unsigned long long)) != cudaSuccess) {
        cudaGetLastError();
        err = "pascal tier: CUDA device " + std::to_string(device) + " did not start";
        return false;
    }
    CUcontext ctx = nullptr;
    if (!dk(d.ctx_get(&ctx), "cuCtxGetCurrent", err)) return false;
    ctx_ = ctx;
    CUmodule m = nullptr;
    if (!dk(d.load(&m, basalt_pascal_cubin), "cuModuleLoadData (the embedded sm_60 cubin)", err)) return false;
    mod_ = m;
    (void) basalt_pascal_cubin_size;
    for (int g : kGuTypes)
        for (int t : kDownTypes) {
            const std::string name = "bp_layer_" + std::to_string(g) + "_" + std::to_string(t);
            CUfunction f = nullptr;
            int per_sm = 0;
            if (!dk(d.get(&f, m, name.c_str()), ("cuModuleGetFunction " + name).c_str(), err) ||
                !dk(d.occupancy(&per_sm, f, kp::kWarps * 32, 0), "cuOccupancyMaxActiveBlocksPerMultiprocessor", err))
                return false;
            if (per_sm < 1) { err = "pascal tier: " + name + " does not fit on an SM"; return false; }
            fn_[g][t] = f;
            blocks_[g][t] = per_sm * nsm;   // the phases' grid barriers need every block resident
        }
    return true;
#endif
}

bool PascalKernels::supports(int gu_type, int d_type) const {
    return gu_type >= 0 && gu_type < kTypes && d_type >= 0 && d_type < kTypes && fn_[gu_type][d_type] != nullptr;
}

bool PascalKernels::layer(int gu_type, int d_type, kp::LayerArgs a, cudaStream_t s, std::string& err) {
    if (!supports(gu_type, d_type)) {
        err = "pascal tier: no layer kernel for gate/up type " + std::to_string(gu_type) + " with down type " +
              std::to_string(d_type);
        return false;
    }
    if (a.groups <= 0 || a.entries <= 0) return true;
    const Driver& d = driver();
    CUcontext cur = nullptr;
    if (!dk(d.ctx_get(&cur), "cuCtxGetCurrent", err)) return false;
    if (cur != (CUcontext) ctx_ && !dk(d.ctx_set((CUcontext) ctx_), "cuCtxSetCurrent", err)) return false;
    const int blocks = blocks_[gu_type][d_type];
    a.bar = bar_;
    a.bar_base = bar_base_;
    void* params[] = {&a};
    if (!dk(d.launch((CUfunction) fn_[gu_type][d_type], (unsigned) blocks, 1, 1, kp::kWarps * 32, 1, 1, 0,
                     (CUstream) s, params, nullptr),
            "bp_layer", err))
        return false;
    bar_base_ += (unsigned long long) kp::kBarriers * (unsigned long long) blocks;   // the launch's arrivals
    return true;
}

}  // namespace basalt::core
