# CUDA driver API smoke test (no toolkit needed): list devices, then on each one allocate 1 GiB, fill it, copy a
# slice back and check it. Exits non-zero if any step fails. Runs on Windows (nvcuda.dll) and Linux/WSL (libcuda).
import ctypes, sys

cuda = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35
TEST_BYTES = 1 << 30
FILL = 0xA5
CHECK_BYTES = 1 << 20


def check(status, what):
    if status != 0:
        name = ctypes.c_char_p()
        cuda.cuGetErrorName(status, ctypes.byref(name))
        raise RuntimeError(f"{what} failed: {status} {name.value.decode() if name.value else ''}")


def attribute(dev, attr):
    value = ctypes.c_int()
    check(cuda.cuDeviceGetAttribute(ctypes.byref(value), attr, dev), "cuDeviceGetAttribute")
    return value.value


def test_device(ordinal):
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
    name = ctypes.create_string_buffer(256)
    check(cuda.cuDeviceGetName(name, 256, dev), "cuDeviceGetName")
    total = ctypes.c_size_t()
    check(cuda.cuDeviceTotalMem_v2(ctypes.byref(total), dev), "cuDeviceTotalMem")
    cc = (attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR), attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))
    ctx = ctypes.c_void_p()
    check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
    check(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    free = ctypes.c_size_t()
    check(cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
    ptr = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(TEST_BYTES)), "cuMemAlloc 1 GiB")
    check(cuda.cuMemsetD8_v2(ptr, FILL, ctypes.c_size_t(TEST_BYTES)), "cuMemsetD8")
    host = (ctypes.c_ubyte * CHECK_BYTES)()
    check(cuda.cuMemcpyDtoH_v2(host, ctypes.c_uint64(ptr.value + TEST_BYTES - CHECK_BYTES), ctypes.c_size_t(CHECK_BYTES)), "cuMemcpyDtoH")
    ok = all(b == FILL for b in host)
    check(cuda.cuMemFree_v2(ptr), "cuMemFree")
    check(cuda.cuDevicePrimaryCtxRelease_v2(dev), "cuDevicePrimaryCtxRelease")
    return name.value.decode(), cc, attribute(dev, CU_DEVICE_ATTRIBUTE_TCC_DRIVER), total.value, free.value, ok


def main():
    check(cuda.cuInit(0), "cuInit")
    version = ctypes.c_int()
    check(cuda.cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion")
    count = ctypes.c_int()
    check(cuda.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    print(f"CUDA driver {version.value // 1000}.{version.value % 1000 // 10}, {count.value} devices")
    failed = False
    for i in range(count.value):
        name, cc, tcc, total, free, ok = test_device(i)
        print(f"  {i}: {name} sm_{cc[0]}{cc[1]} tcc={tcc} total={total >> 20} MiB free={free >> 20} MiB "
              f"1GiB alloc+fill+readback={'OK' if ok else 'MISMATCH'}")
        failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
