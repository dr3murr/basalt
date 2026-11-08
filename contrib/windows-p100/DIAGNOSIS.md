# Windows: why the P100 runs in MCDM, not TCC (and the RTX 5080's Code 43 on the way)

> Written with AI coding agents and measured on one machine (RTX 5080 + Tesla P100, Ryzen 9 9950X3D, 64 GB DDR5); not
> reviewed by a kernel/GPU engineer. Use at your own risk. The setup described here is a modified driver install (an
> INF-edited, re-signed Data Center 582.99 package, see [README.md](README.md)); nothing was tested on a stock driver
> install.

Found on the test machine (Windows 11 23H2, Hyper-V/WSL2 enabled, so the hypervisor manages the IOMMU) on 2026-11-04
and 2026-11-05, with the kit's scripts as they were then.

## RTX 5080 Code 43 after the first reboot: cause and fix

**Cause:** the first version of the P100 script (then `3b_wddm_keys.ps1`, now `3b_p100_driver_model.ps1`) ran
`New-Item -Path HKLM:\...\Services\nvlddmkm\Parameters -Force`. On an existing registry key, `New-Item -Force`
deletes all of the key's values (tested on a scratch HKCU key). That removed the INF's `DmaRemappingCompatible = 3`
(and `WDDMv22EnableSmoothTransition = 0`).

**Effect (traced with DTrace and the kernel decompiled in IDA):**

1. `nt!PipDmgGetDriverDmarCompatLevel` reads `Services\<driver>\Parameters\DmaRemappingCompatible`
   (0 -> level 1, 1 or 3 -> level 2, 2 -> level 3, missing -> 0). `PipDmgSaveDeviceDmarPolicy` stores the result as
   `DEVPKEY_Device_DmaRemappingPolicy` and sets the "driver manages DMA remapping" flag only for policy 2.
2. With the value missing the policy was 0, so before the start PnP's DMA guard unblocked the GPU into a default
   IOMMU domain (`PiDmaGuardProcessPreStart -> PiIommuUnblockDevice -> HalpIommuUnblockDevice`).
3. dxgkrnl then unblocked it again: 582.99 reports SysMm caps `0x5` (no GPU-VA IOMMU bit), domain types `0x3`, so
   `SmmUnblockDevice -> SmmIommuSwitchToPassthrough -> SmmSwapDomains` attached a passthrough domain assuming none.
4. The hypervisor (running for Hyper-V/WSL2) refused: `HvlDmaAttachDeviceDomain` returned `0xC0350066` =
   `HV_STATUS_DEVICE_ALREADY_IN_DOMAIN` (0x66, Linux `include/hyperv/hvgdk_mini.h`). `DpiFdoStartNonLdaAdapter`
   failed before nvlddmkm's `DxgkDdiStartDevice` was ever called; dxgkrnl invalidated the device: Code 43.

The P100 was unaffected: in TCC it never goes through dxgkrnl's adapter start. That same fact is what broke it once
the value was back (next section). Before the reboot the 5080 had started while the value was still there, which is
why it worked then.

**Fix:** `DmaRemappingCompatible = 3` and `WDDMv22EnableSmoothTransition = 0` restored; every script now creates a
key only if it is missing. After a disable/enable of the 5080 its `DmaRemappingPolicy` reads 2 and it runs in WDDM.
If this ever recurs, check that value first:
`(Get-ItemProperty HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Parameters).DmaRemappingCompatible` must be 3.

Same bug, other key: the first `1_backup.ps1` used `New-Item -Force` on
`HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate`, which would have removed any other Windows Update policy
values or subkeys there (the test machine's local policy had none).

## Tesla P100 Code 10 on every boot after that fix: cause and fix

**Symptom:** P100 `CM_PROB_FAILED_START`, ProblemStatus `0xC000009A` (STATUS_INSUFFICIENT_RESOURCES), "This device
cannot start (Code 10)". The 5080 worked.

**Cause:** `DmaRemappingCompatible = 3` is per service, so it gives both nvlddmkm GPUs DMA policy 2: PnP's DMA guard
does not open the device's IOMMU domain before the start (`PiIommuUnblockDevice` returns early) and leaves that to the
driver. For a WDDM or MCDM adapter dxgkrnl does it (`SmmUnblockDevice`). A TCC adapter never goes through dxgkrnl,
so the P100's DMA stayed blocked: nvlddmkm ran its GPU init for about 4 s, every kernel call it made succeeded (BAR0
mapped, interrupts connected, contiguous memory allocated), and then it gave up with `0xC000009A`.
After one such failed start the P100 fails every later start until a reboot, under any policy [INFERENCE: a GPU
state that disable/enable does not reset]. That hid the cause: every mid-boot test after a failed boot start failed.

**Evidence:**

| Boot | P100 policy at its first start | Result |
|---|---|---|
| two boots with the value wiped by the Code 43 bug | 0 | P100 OK in TCC, 5080 Code 43 |
| every boot with the value restored (catalog full or INF-only, 5080 enabled or disabled, cold boot) | 2 | P100 Code 10 |
| test boot: value removed, then restored for the 5080 mid-boot | 0 | P100 OK at boot, after its restart, and after the 5080 ran |
| value 3, P100 in MCDM (`AdapterType = 4`) | 2 | both OK |

Under policy 0 the DMA guard attaches the P100 itself (`HalpIommuUnblockDevice -> HvlDmaAttachDeviceDomain`, traced);
under policy 2 it does not. Not the cause, each tested directly: the INF-only (EAC-style) catalog, `Parameters\FTS`
and the other Parameters values, start order, the 5080 running first, a cold boot, and an unrelated RAM-disk driver
that was failing on that machine (Code 12).

**Fix:** run the P100 in MCDM (`3b_p100_driver_model.ps1`, `AdapterType = 4`), keeping NVIDIA's
`DmaRemappingCompatible = 3`. The P100 then starts through dxgkrnl like the 5080. CUDA works on both GPUs natively
and in WSL; DXGI lists only the 5080. Every package install resets the P100 to TCC, so steps 3 and 3e run 3b at the
end.

## Tools used

DTrace for Windows (fbt probes on nt/dxgkrnl, kernel calls from nvlddmkm on the `PnpStartDevice` thread), ETW
(`Microsoft-Windows-Kernel-PnP`, `Microsoft-Windows-DxgKrnl`, NVIDIA's binary-only provider
`{10DE00F0-50FE-4D0E-A83D-BAE3F58C94AA}`), and IDA on `ntoskrnl.exe` 10.0.22621.5624 for the DMA-guard code. The
DTrace scripts are not included. DTrace was uninstalled afterwards; it needs `bcdedit /set {current} dtrace on` and a
reboot to run.
