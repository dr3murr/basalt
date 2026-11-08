# RTX 5080 + Tesla P100 on Windows 11: one driver for both, and Basalt with the Pascal tier

> **Disclosure.** Written with AI coding agents and measured on one machine (RTX 5080 + Tesla P100, Ryzen 9 9950X3D,
> 64 GB DDR5); not reviewed by a kernel/GPU engineer. Use at your own risk.
>
> **Driver.** Running the two cards together needs a modified driver setup. On Windows that is NVIDIA's Data Center
> 582.99 package with two lines added to its INF and its catalog re-signed with a certificate you create and trust
> yourself (on Linux: the proprietary R580 driver plus an undocumented module option, see
> [PASCAL.md](../../PASCAL.md)). Nothing here was tested on a stock driver install.

These scripts got an RTX 5080 (display, WDDM) and a Tesla P100 (compute only, MCDM) working under one NVIDIA driver on
Windows 11 23H2 (build 22631), with Secure Boot on and no test-signing, and built Basalt natively with its Pascal
expert tier. The engine side, the Linux numbers and how the tier works are in [PASCAL.md](../../PASCAL.md). Why the
P100 cannot use TCC here, and the Code 43 hit on the way: [DIAGNOSIS.md](DIAGNOSIS.md).

This folder contains no NVIDIA or Microsoft files. You download the driver package and the Windows SDK/WDK tools
yourself; the scripts edit and sign your copy on your machine.

## What it does

- **Driver:** the Data Center 582.99 package (R580, the newest branch that still supports Pascal; R590 drivers such
  as GeForce 591.59 do not) with two lines added to `Display.Driver\nv_dispwi.inf`: the RTX 5080 (`DEV_2C02`) mapped
  to the section 582.99 uses for the RTX PRO 4500/4000 Blackwell (the same GB203 die), and a P100 line without a
  subsystem ID (`DEV_15F8`; the stock line matches only subsystem 118F). The GeForce and Data Center `nvlddmkm.sys` of
  one version are the same code (every section identical on 582.78; only the checksum and signature differ), so this
  is an INF change only.
- **Signing:** the edited INF breaks the package's catalog. Step 3 or 3e rebuilds `nv_disp.cat`, signs it with a fresh
  local certificate (`signing_cert.ps1`), deletes the certificate's private key and trusts the certificate in
  LocalMachine Root and TrustedPublisher, as NVCleanstall does for its modified drivers. The driver binaries keep
  NVIDIA's embedded signatures: **no test-signing, Secure Boot stays on.** Read
  [What trusting a self-made root certificate means](#what-trusting-a-self-made-root-certificate-means) first.
- **Modes:** the 5080 runs in WDDM (display), the P100 in MCDM (compute only). TCC, the INF's default for the P100,
  cannot start next to the INF's `DmaRemappingCompatible = 3` ([DIAGNOSIS.md](DIAGNOSIS.md)). MCDM goes through
  dxgkrnl like the 5080, so CUDA and WSL see the P100 while DXGI (games) lists only the 5080. No P2P is used or needed.
- **Inference:** Basalt from this branch, built with MSVC 19.44, CUDA 13.3 and CUDA 12.9 (the P100's sm_60 cubin),
  serving Qwen3.8-Flash-Next with the settings of the Linux service.

## What you need

- Windows 11 x64; administrator rights. Between steps 2 and 3 the screen runs on Microsoft Basic Display.
- The NVIDIA Data Center driver 582.99 for Windows 10/11:
  `https://us.download.nvidia.com/tesla/582.99/582.99-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe`
  (723,345,432 bytes, SHA-256 `1c5614b38fb015e5c1218da22c41caf103800d9c1b963b531dda5f0e88ce09d8`; check with
  `Get-FileHash`). Extract it with 7-Zip: `7z x 582.99-...exe -oC:\nv58299`; the folder the scripts call `-DriverDir`
  is then `C:\nv58299\Display.Driver`.
- The Windows SDK (`signtool.exe`, `makecat.exe`) and, for step 3, the Windows Driver Kit (`Inf2Cat.exe`). Step 3e
  needs only the SDK. The scripts look under `C:\Program Files (x86)\Windows Kits\10\bin`; step 3 also takes
  `-Inf2Cat` and `-Signtool` paths.
- Visual Studio 2022 or 2026 with the C++ workload, CMake 3.24 or newer, Ninja, Python 3.10 or newer. Step 5 adds the
  MSVC 14.44 toolset and the CUDA toolkit parts.
- The P100 itself needs cooling (it has no fan) and an 8-pin EPS power cable ([PASCAL.md](../../PASCAL.md#hardware)).

## The INF change

`scripts\patch_inf.ps1 -DriverDir C:\nv58299\Display.Driver` makes it for you and checks the result. It inserts these
lines into `nv_dispwi.inf` (ASCII, CRLF line ends):

After the line that starts with `%NVIDIA_DEV.2C34.2052.10DE%` (the RTX PRO 4000 Blackwell, near the end of the
section `[NVIDIA_Devices.NTamd64.10.0...17134]`):

```
%NVIDIA_DEV.2C02%           = Section094, PCI\VEN_10DE&DEV_2C02
%NVIDIA_DEV.15F8%           = Section026, PCI\VEN_10DE&DEV_15F8
```

After the line that starts with `NVIDIA_DEV.2C34.2052.10DE =` in `[Strings]`:

```
NVIDIA_DEV.2C02 = "NVIDIA GeForce RTX 5080"
NVIDIA_DEV.15F8 = "NVIDIA Tesla P100-PCIE-16GB"
```

`Section094` gives the 5080 the registry values an RTX PRO 4500/4000 Blackwell gets: WDDM, plus two graphics
power-gating values (`RMLpwrGrRgIdleThresholdUs`, `RmLpwrGrRgWar5291421`) and a stereo setting that the 5080's own
section in the GeForce 581.80 INF does not set; nothing else kernel-relevant differs (static comparison of the two
INFs). `Section026` is the stock P100 section (TCC by default, which step 3b changes to MCDM).

The script pads the model lines with spaces to the INF's own columns, so its output is byte-identical to the file
the test machine installed: the stock INF has SHA-256 `080b2285660112ffdeb2d4fbd35886ca9a69febb725074f5349d8e18e0c66e4d`,
the patched one `c0a840e3674b4c64f0a9de85afd7c958351bea89c609fef9b7f4cc47d308d6fa`. It refuses any other input
unless you pass `-Force`, and saves the original as `nv_dispwi.inf.orig` next to the `Display.Driver` folder. Editing
by hand works too (the padding does not matter to Windows), but then the hash check above does not apply.

## Steps

Run each script from an elevated Windows PowerShell 5.1:
`powershell -ExecutionPolicy Bypass -File scripts\<name>.ps1 <parameters>`. The scripts keep their state (driver
backup, certificate thumbprints, logs) in `-StateDir`, by default `%ProgramData%\p100-win`; pass the same value to
every script if you change it.

1. `1_backup.ps1`: exports the installed NVIDIA display driver package(s) to `<StateDir>\driver-backup`, blocks
   drivers from Windows Update (`ExcludeWUDriversInQualityUpdate = 1`), and makes a restore point.
2. `2_remove_current.ps1`: removes the installed NVIDIA display packages. An R590 package (no Pascal support) would,
   as a WHQL driver, outrank the self-signed one for the 5080. **Reboot.**
   Then `patch_inf.ps1 -DriverDir <Display.Driver>` (no administrator rights needed; any time before step 3).
3. `3_install_r580.ps1 -DriverDir <Display.Driver>`: full catalog (Inf2Cat), new certificate, signature, private key
   deleted, certificate trusted, `pnputil /add-driver ... /install`, then step 3b. **Reboot.**
   `start_unattended.ps1 -DriverDir <Display.Driver>` runs step 3 as a SYSTEM task instead (it survives the display
   restarting under the terminal), reboots by itself and writes `4_verify.ps1`'s output to `<StateDir>\verify.log`
   after the boot.
   - `3b_p100_driver_model.ps1 [-DriverModel MCDM|WDDM]` (run by 3 and 3e): writes the P100's `AdapterType`
     (4 = MCDM, the default; 1 = WDDM) and the Tesla keys it was tested with (`AllowWddmOnTesla` in
     `Services\nvlddmkm\Parameters` and on the P100's adapter key, `GridLicensedFeatures = 7`, `EnableMsHybrid = 1`).
     Every package install resets the P100 to TCC, so it has to run after each one. Reboot afterwards.
   - `3e_reinstall_eac.ps1 -DriverDir <Display.Driver> [-AutoAccept]`: instead of or after step 3. NVCleanstall's
     Easy-Anti-Cheat mode: the catalog lists only the INFs (makecat), so every binary is validated by its own embedded
     NVIDIA/Microsoft signature and no driver file depends on your certificate. Signs with a new certificate,
     reinstalls, removes the certificates of earlier runs, runs 3b. Windows shows the "can't verify the publisher"
     prompt once; `-AutoAccept` presses "Install this driver software anyway". **Reboot.** The test machine ran this
     catalog when the numbers below were measured.
4. `4_verify.ps1`: expect both GPUs on 582.99, 5080 = WDDM, P100 = MCDM, DMA policy 2 on both, and `cuda_smoke.py`
   passing on both (it allocates, fills and reads back 1 GiB per GPU; needs only Python). If the P100 shows Code 10
   (`0xC000009A`), it started in TCC: run 3b and reboot.
5. `5_install_toolchain.ps1 [-Downloads <dir>]`: MSVC v143 (19.44) added to Visual Studio, CUDA 12.9 (nvcc, cudart)
   and CUDA 13.3 (nvcc, crt, nvvm, thrust/CCCL, cudart, cuBLAS, nvJitLink), toolkit components only; the CUDA
   installers' bundled driver would replace the edited package. CUDA 13 cannot compile sm_60, so 12.9 builds the P100
   tier's cubin, as on Linux. Fails if the display driver is no longer 582.99.
6. `6_build_basalt.ps1 [-Src <checkout>] [-Build <dir>] [-Jobs N] [-TestJobs N]`: configures with Ninja from a v143
   developer shell, builds (the programs land in `<Src>\bin`) and runs the CPU tests. 61 of 63 pass on the test
   machine, so the script ends with "CPU tests failed" after a complete build
   ([Known test failures](#known-test-failures)). Then install the server's Python packages where `basalt serve`
   looks for them: `python -m venv .venv` and `.venv\Scripts\python -m pip install -r requirements.txt` in the
   checkout.
7. `7_get_pack.ps1 -Dest <dir>`: downloads `Qwen3.8-FN-NVFP4-DQ6_K.basalt` (131.7 GB, resumable: rerun to continue)
   from Hugging Face and checks its size and SHA-256.
8. `8_run_basalt.ps1 -Pack <dir>\Qwen3.8-FN-NVFP4-DQ6_K.basalt`: writes `%USERPROFILE%\.config\basalt\flashnext.json`
   (the Linux service's settings) on first use and runs `basalt serve` on `http://127.0.0.1:8090`. Run it elevated
   for large pages.

## What trusting a self-made root certificate means

Steps 3 and 3e create a self-signed certificate and add it to **LocalMachine\Root** (trusted root certification
authorities) and **LocalMachine\TrustedPublisher**. That is what lets Windows install a driver package whose catalog
you signed yourself without test-signing. It also means:

- **Windows trusts anything signed with that key, machine-wide, for what the certificate allows:** code signing and
  driver package (catalog) signing (its extended key usages: Code Signing, Windows Hardware Driver Verification,
  Windows Hardware Driver Extended Verification). Anyone holding the private key could sign software or a driver
  package that this machine then treats as coming from a trusted publisher, without any prompt. It does not make
  kernel code load: with Secure Boot on, Windows loads kernel drivers only with Microsoft-issued signatures, and none
  of NVIDIA's binaries are re-signed.
- **The scripts make the key hard to misuse:** it is generated as non-exportable in LocalMachine\My, used once to sign
  `nv_disp.cat`, and deleted right after, also when signing fails (`Remove-Item ... -DeleteKey` in a `finally`).
  Nothing can sign with it afterwards. NVCleanstall, by comparison, writes its private key to a passwordless .pfx
  while it signs and never removes its root certificate (static analysis of NVCleanstall 1.19.0). The risk left is
  the few seconds in which the key exists, and an attacker who can read it then already has administrator rights.
- **The certificate stays trusted for 5 years unless you remove it.** Each run of step 3 or 3e makes a new one and
  records its thumbprint in `<StateDir>\cert-thumbprint.txt`; 3e removes the ones it finds there from earlier runs.
  See them in `certlm.msc` under Trusted Root Certification Authorities and Trusted Publishers.
- **Subject name.** The default subject is `CN=p100-win local driver catalog signer (self-signed)`. The test machine
  used NVCleanstall's subject string instead (`-Subject 'CN=NVCleanstall Driver Modding Authority,
  O="https://www.techpowerup.com"'`), so that tools which already know NVCleanstall-modded drivers would see a
  familiar issuer; whether that matters for any anti-cheat was not tested. Either way the certificate is made on your
  machine and has nothing to do with TechPowerUp.
- **Updates.** The package is signed by you, not by Microsoft's WHQL program; Windows Update could offer a WHQL
  GeForce driver for the 5080 that drops the P100. Step 1 blocks driver updates through Windows Update for that
  reason.

## Rollback

`rollback.ps1 [-StateDir <dir>]` (elevated), then reboot. It:

1. removes every installed NVIDIA display package (`pnputil /delete-driver ... /uninstall /force`);
2. removes every certificate listed in `<StateDir>\cert-thumbprint.txt` from LocalMachine Root, TrustedPublisher and
   My (and its private key, if one is left), then from that file; if the file is empty it tells you to check
   `certlm.msc` by hand;
3. reinstalls the driver package(s) step 1 exported to `<StateDir>\driver-backup`.

It does not undo: the Windows Update driver block (delete
`HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\ExcludeWUDriversInQualityUpdate` and set
`HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\DriverSearching\SearchOrderConfig` back to 1), the values step 3b
wrote (`AllowWddmOnTesla` in `Services\nvlddmkm\Parameters`; `AdapterType`, `AllowWddmOnTesla`,
`GridLicensedFeatures`, `EnableMsHybrid` on the P100's adapter key), and the restore point. If your previous driver
was R590 or newer (on the test machine: GeForce 591.59), the P100 has no driver again after the rollback, because
those drivers do not support Pascal. The restore point from step 1 is a second way back.

## Measured numbers

`basalt-bench -t conv,struct,prose -r 2 --seed 1`, `--gpu 0,1` unless noted, decode tokens/s. "f14e879" is f14e879
plus 086814a, a compile fix for MSVC. Sources: the kit's measurement table of 2026-11-04/05 and, for 6e4cc31, the
three runs' logs and the commit message ([PASCAL.md](../../PASCAL.md#sources)).

| setup | conv | struct | prose |
|---|---:|---:|---:|
| Linux, 5080 + P100 (11f2e5b; f14e879 is 1.5-5% faster per window) | 122 | 201 | 122 |
| Linux, 5080 alone | 82 | 83 | 80 |
| Windows, f14e879 automatic plan (P100 mirrors RAM), 48 GB free | 66.6 (run 1), then stopped by the RAM guard | | |
| Windows, `--peer-exclusive`, 48 GB free (2.9 GiB of experts on the SSD, swaps deferred) | 108 | 89 | 101 |
| Windows, `--peer-exclusive --resident-budget-gib 32`, 48 GB free | 68 ± 13 | 151 ± 8 | 71 |
| Windows, `--peer-exclusive`, 58 GB free (apps closed; nothing on the SSD) | 91 ± 3 | 172 ± 0 | 89 ± 3 |
| Windows, `--peer-exclusive`, 55 GB free, 6e4cc31 (swaps through pinned slots; mean of 3 runs) | 116 | 187 | 112 |
| Windows, 5080 alone (`--gpu 0`), 58 GB free (8.1 GiB of experts on the SSD, swaps deferred) | 70 ± 0 | 66 ± 1 | 70 ± 1 |

Through `basalt serve` (f14e879, 58 GB free): 80 t/s on the first, cold request; a 676-token JSON reply in 7.3 s end
to end. Free RAM decides most of it: close other programs before starting the server. The Linux rows were measured
with the P2P-patched 580.173.02 driver loaded; Basalt does not use P2P ([PASCAL.md](../../PASCAL.md#driver-setup-linux)).

## Known risks

- Windows caps page-locked host memory: ~15.4 GiB per allocation and 28 GiB in total were measured on the test
  machine, for either GPU and in WSL too. f14e879's resident RAM copy asks for the whole arena at once and, when
  refused, pins none of it ("page-locking refused ... pinned 0.00 GiB"). Every expert swap was then a staged copy that
  held the decode loop and the P100's launches; 6e4cc31 sends those copies through small pinned slots, which closed
  most of the gap to Linux. What is left (~2 ms per conv window): the copier's memcpy (~10 ms per window on one core,
  beside the window), the P100 refills' memcpy, no PCIe reads of experts from RAM (the GPU cannot read pageable
  memory), and 0.8 GiB less room on the P100 under MCDM (14.2 vs 15.0 GiB).
- Pinning part of the RAM copy (16-20 GiB) might win back about 1 ms per window, but that sits at the Windows
  pinning limit, where an earlier test got the driver stuck. Not done.
- The 5080 drives the display, so it is subject to TDR (2 s by default). Raise `TdrDelay` under
  `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` if long kernels get reset.
- Untested: whether MCDM needs the Tesla keys 3b sets (tested only with them); any anti-cheat; any GPU pair other
  than this RTX 5080 and Tesla P100-PCIE-16GB; quality (KLD/PPL) on Windows.
- `DmaRemappingCompatible = 3` under `Services\nvlddmkm\Parameters` must stay; scripts that create registry keys with
  `New-Item -Force` on an existing key wipe its values (that caused the 5080's Code 43, [DIAGNOSIS.md](DIAGNOSIS.md)).

## Known test failures

On the test machine 61 of 63 CPU tests pass:

- `iq_avx512_test` compares against hashes frozen from the GCC build, and MSVC's results differ in the last bits
  (max relative error vs double 3.6e-6 to 7.0e-6; `docs/precision/builds.md` covers MSVC vs GCC parity).
- `platform_system_test` fails `keep_full_speed` because power throttling is off system-wide on that machine
  (`PowerThrottlingOff = 1`), so Windows rejects the per-process opt-out with `ERROR_INVALID_FUNCTION`.

How many tests are registered depends on Python: with a Python 3 that has numpy and gguf-py at configure time (the
test machine's checkout has numpy in its `.venv`), 63 CPU tests are registered. Without it `iq_parity_fixtures`,
`iq_parity` and `convert_tests` are left out: a fresh clone of this branch built on 2026-11-08 with
`6_build_basalt.ps1 -Jobs 4 -TestJobs 2` (MSVC 19.44.35228, CUDA 13.3.73, the cubin from CUDA 12.9) registered 60
CPU tests, passed 58 and failed the same two; `auto_plan_test`, which covers b1049ff's rule, passed.

## Credits and prior art

- **Signing method:** [NVCleanstall](https://www.techpowerup.com/download/techpowerup-nvcleanstall/) (TechPowerUp):
  an edited INF, a catalog re-signed with a fresh self-signed certificate trusted in LocalMachine Root and
  TrustedPublisher, the binaries keeping NVIDIA's signatures, and the INF-only catalog of its Easy-Anti-Cheat mode.
  The kit's scripts reimplement that method from a static analysis of NVCleanstall 1.19.0.
- **The same INF idea for GeForce + Tesla:** [Z0ntiQ/nv-dual-gpu-driver-patch](https://github.com/Z0ntiQ/nv-dual-gpu-driver-patch)
  adds a GeForce device ID to a same-chip donor section of the Data Center INF and re-signs the catalog (verified there
  with a V100 + RTX 2060 on 582.78); unlike this kit it turns test-signing on for the install.
- **The Tesla registry keys** step 3b writes (`AdapterType`, `GridLicensedFeatures = 7`, `EnableMsHybrid = 1`,
  `AllowWddmOnTesla`) come from the community: the Level1Techs thread
  ["Gaming on my Tesla"](https://forum.level1techs.com/t/gaming-on-my-tesla-more-likely-than-you-think/171185) and
  [TeslaP40_wddm](https://github.com/Ilan12346-maya/TeslaP40_wddm) (whose `enable_wddm.bat` writes them).
- **MCDM on a Pascal Tesla** was documented before for a Tesla P40 under WSL2:
  [schmared/tesla-p40-wsl2-cuda-llamacpp](https://github.com/schmared/tesla-p40-wsl2-cuda-llamacpp) (it switches with
  `nvidia-smi -dm 2`; whether the extra keys 3b writes are needed for MCDM was not tested here).
- Not found in earlier public write-ups (searched 2026-11-08): an RTX 50 card and a Pascal Tesla on one package with
  no test-signing at any step, and the cause of the P100's TCC Code 10 next to `DmaRemappingCompatible = 3`
  ([DIAGNOSIS.md](DIAGNOSIS.md)).

## Files

Run on the test machine (2026-11-04/05) in the form they had then, with paths fixed to that machine; for publication
the paths became parameters (`-DriverDir`, `-StateDir`, `-Src`, `-Pack`, ...) and the changes below were made. The
changed scripts were syntax-checked; `patch_inf.ps1` was checked against the stock INF, and `6_build_basalt.ps1`
built this branch from a fresh clone. The driver scripts were not run again in this form.

| file | what it does | change for publication |
|---|---|---|
| `1_backup.ps1` | driver backup, Windows Update block, restore point | backup in `<StateDir>\driver-backup` |
| `2_remove_current.ps1` | removes the installed NVIDIA display packages | paths |
| `patch_inf.ps1` | the INF change, byte-exact, with hash checks | new (the test machine's INF was edited with an equivalent Python script) |
| `signing_cert.ps1` | certificate helpers (dot-sourced) | thumbprint file and subject as parameters; neutral default subject; SDK/WDK tool lookup |
| `3_install_r580.ps1` | full catalog (Inf2Cat), sign, install, 3b | `-DriverDir`; Inf2Cat/signtool from the Windows Kits folder instead of a bundled copy; private key deleted in a `finally` |
| `3b_p100_driver_model.ps1` | P100 to MCDM (or WDDM) | none |
| `3e_reinstall_eac.ps1` | INF-only catalog (makecat), sign, reinstall, 3b | `-DriverDir`, `-StateDir`; private key deleted in a `finally` |
| `4_verify.ps1` | device state, driver models, DMA policy, CUDA smoke test | none |
| `cuda_smoke.py` | 1 GiB allocate/fill/read back per GPU through the driver API | none |
| `5_install_toolchain.ps1` | MSVC 14.44, CUDA 12.9 and 13.3 toolkit parts | default download folder in `%TEMP%` |
| `6_build_basalt.ps1` | configure, build, CPU tests | `-Src` defaults to this checkout; `-Build`, `-Jobs`, `-TestJobs` added |
| `7_get_pack.ps1` | downloads and checks the model pack | replaces `7_prepare_pack.ps1`, which also reformatted that machine's Linux partition as NTFS |
| `8_run_basalt.ps1` | writes the config, runs `basalt serve` | `-Pack` required |
| `rollback.ps1` | undoes steps 3/3e (see above) | `-StateDir`; warns when no thumbprints are recorded |
| `start_unattended.ps1`, `run_unattended.ps1` | step 3 as a SYSTEM task with automatic reboot and verify log | `-DriverDir`, `-StateDir` |
| [DIAGNOSIS.md](DIAGNOSIS.md) | why TCC fails next to `DmaRemappingCompatible = 3`; the Code 43 on the way | machine-specific paths removed |
