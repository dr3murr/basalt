# A Tesla P100 (Pascal, sm_60) expert tier for Basalt

> **Disclosure.** Written with AI coding agents and measured on one machine (RTX 5080 + Tesla P100, Ryzen 9 9950X3D,
> 64 GB DDR5); not reviewed by a kernel/GPU engineer. Use at your own risk.
>
> **Driver.** Running the two cards together needs a modified driver setup. Linux: NVIDIA's proprietary R580 kernel
> modules plus an undocumented module option. Windows: NVIDIA's Data Center 582.99 package with an edited INF and a
> catalog re-signed with a certificate you create and trust yourself. Nothing here was tested on a stock driver
> install.

This is branch `pascal-tier` of [dr3murr/basalt](https://github.com/dr3murr/basalt), a fork of
[jesdga95/basalt](https://github.com/jesdga95/basalt). It branches off upstream commit `9595ae0` (2026-11-03) and is
not rebased onto later upstream work. Upstream has not reviewed any of it. Everything specific to this fork is
described here; the rest of the repository is upstream Basalt.

The *expert tier* is upstream Basalt's mechanism for a second Blackwell card, which Basalt took over from Strata's
helper caches and peer tier ([Prior art and credits](#prior-art-and-credits)). Upstream Basalt runs only on sm_120
and stops on any other card. This fork makes the tier run on a Pascal card, here a Tesla P100 16 GB, next to a
Blackwell main card (here an RTX 5080 16 GB): the P100 holds 15 GiB of experts and computes their rows in every decode
window with kernels of its own, so the CPU computes far fewer experts. The fork also includes a Windows kit that gets
both cards running under one driver ([contrib/windows-p100/](contrib/windows-p100/)).

Sources are given in brackets after each number and listed at the end ([Sources](#sources)).

## Contents

- [Results in one table](#results-in-one-table)
- [Hardware](#hardware)
- [How the Pascal tier works](#how-the-pascal-tier-works)
- [Driver setup: Linux](#driver-setup-linux)
- [Driver setup: Windows](#driver-setup-windows)
- [Building](#building)
- [Running](#running)
- [Results](#results)
- [Where the time goes](#where-the-time-goes)
- [Ceilings](#ceilings)
- [Tried and dropped](#tried-and-dropped)
- [Caveats](#caveats)
- [Reproducing](#reproducing)
- [Commits](#commits)
- [Related work](#related-work)
- [Prior art and credits](#prior-art-and-credits)
- [Sources](#sources)

## Results in one table

Qwen3.8-Flash-Next, pack `Qwen3.8-FN-NVFP4-DQ6_K.basalt` (NVIDIA's NVFP4 experts). Decode tokens/s from
`basalt-bench -t conv,struct,prose -r 2 --seed 1` (Basalt's three bench prompts: a conversation, structured JSON
output, prose).

| setup | conv | struct | prose | source |
|---|---:|---:|---:|---|
| Linux, RTX 5080 alone (`--gpu 0`) | 82 | 83 | 80 | [R] |
| Linux, RTX 5080 + P100 (commit 11f2e5b) | 122 | 201 | 122 | [R], [C:11f2e5b] |
| Windows 11, RTX 5080 alone (`--gpu 0`, f14e879, 58 GB RAM free) | 70 | 66 | 70 | [K] |
| Windows 11, RTX 5080 + P100 (6e4cc31, `--peer-exclusive`, 55 GB RAM free, mean of 3 runs) | 116 | 187 | 112 | [K], [L], [C:6e4cc31] |

On Linux the P100 adds +49% (conv), +142% (struct) and +53% (prose) [R]. Windows with the P100 reaches 95%, 93% and
92% of the Linux figures (computed from the table), but those two rows come from different builds (Windows 6e4cc31,
Linux 11f2e5b). Struct, whose drafts are accepted almost every time (98% in the Windows logs [L]), gains the most. The
Linux struct figure of 201 is the bench table's number, which reads high on struct. Measured as time per decode window
on comparable builds (Windows 6e4cc31, Linux 98f6932), Windows takes about 11% longer on conv (23.80 vs 21.46 ms) and
7% longer on prose (19.54 vs 18.23 ms), and the two are equal on struct (41.6 vs 41.65 ms) [S:0fcf3c47], [L], [R].

The Linux rows were measured while the P2P-patched 580.173.02 driver was loaded; Basalt does not use P2P (see
[Driver setup: Linux](#driver-setup-linux)). On the stock 580.196.07 package with the coexistence option below, a
served request decoded at 124.9 t/s [S:ba873ef6] and the 32 GB RAM runs further down ran on it [R].

## Hardware

| part | detail | source |
|---|---|---|
| main card | NVIDIA GeForce RTX 5080 16 GB (GB203, sm_120), drives the display, power limit 360 W (stock) | [S:3bd91e10], [L] |
| expert tier | NVIDIA Tesla P100-PCIE-16GB (GP100, sm_60), passively cooled data-center card | |
| P100 extras | a fan and shroud ($15) and an EPS power splitter cable ($5); the card cost $80 | [S:d29b4701] |
| P100 slot | PCIe 3.0 x4, behind the X870E chipset (shares its uplink with Wi-Fi, 2.5 GbE, USB and an SSD) | [S:3bd91e10] |
| P100 clocks | application clock 1189 MHz of 1328 MHz maximum (left at the default) | [S:3bd91e10] |
| CPU, RAM | AMD Ryzen 9 9950X3D (16 cores), 64 GB DDR5 | |
| model disk | NVMe SSD on CPU lanes | [S:3bd91e10] |
| OS | Ubuntu 26.04 LTS (kernel 7.0.0-31-generic) and Windows 11 23H2 (build 22631), dual boot | |

The P100 needs case airflow it does not bring itself (no fan of its own) and an 8-pin EPS (CPU-style) power
connector, not a PCIe one.

Where the 67.8 GiB of weights sat on Linux in the served configuration (int8 KV cache, resident experts): RTX 5080
13.2 GiB, P100 15.0 GiB, system RAM 39.5 GiB, page-locked [S:1828651c]. The pack's n-gram table (~51 GiB) stays on
the SSD and is read a few KB per token [R], [S:1828651c].

## How the Pascal tier works

### Background: Basalt's expert tiers

Basalt runs a mixture-of-experts model whose experts do not fit in VRAM. The main card holds the dense weights, the
KV cache, the drafter and a cache of the hottest experts; the rest are in pinned system RAM and are computed by a CPU
thread pool. A second card can be an *expert tier*: it holds more experts and computes their rows in every decode
window. The tiers follow the conversation: periodic *swap rounds* move experts that are routed often into VRAM and
cold ones out ([docs/configuration/gpus.md](docs/configuration/gpus.md),
[docs/configuration/adaptive.md](docs/configuration/adaptive.md)).

Each layer of a decode window is a handoff: the main card runs attention, the projections, the shared expert and the
router; the host then gets the routing, the CPU pool computes its experts and the tier card computes its own; then the
main card combines the rows and continues. Upstream, a tier card runs the main card's sm_120 kernels.

### What the fork adds

The engine is built for sm_120 only, by CUDA 13. CUDA 13 compiles nothing older than sm_75, so none of Basalt's
kernels can run on a P100. The Pascal tier keeps everything upstream's tier does (its slots, fills, swaps, host rows,
the runtime API) and replaces only the kernels:

- **An embedded CUDA 12 cubin.** `src/kernels/pascal/peer_experts_sm60.cu` is compiled on its own by a CUDA 12.x nvcc
  (`-arch=sm_60 -cubin -O3`, CMake option `BASALT_PASCAL_NVCC`) into `pascal_peer_sm60.cubin`, which
  `cmake/embed_cubin.cmake` turns into a byte array linked into the engine. At start, `PascalKernels::load`
  (`src/core/pascal_peer.cpp`) loads it on the P100 with `cuModuleLoadData`. The driver API entry points come from
  `cudaGetDriverEntryPointByVersion`, so nothing links `libcuda` directly. The only header both builds share,
  `include/basalt/kernels/pascal_peer.hpp`, holds plain data (`LayerArgs`, `Group`).
- **One fused kernel per layer.** `bp_layer_G_D` (G: the gate/up GGML type, D: the down type; the NVFP4 pack uses
  `bp_layer_40_40`) does a whole layer's tier experts in one launch: activations to fp16, gate/up with SwiGLU, the
  intermediate rows to fp16, down, and the result rows to the host. The phases are separated by software grid
  barriers (4 per launch; one counter that is never reset, each launch's targets start at `bar_base`). Grid barriers
  need every block resident, so the kernel is launched with exactly the blocks that fit at once (occupancy times SM
  count, 8 warps per block).
- **Work split.** The host groups the tier's entries by expert, at most 2 entries per group (`kGroupEntries`); each
  group is a 16-byte record: the expert's blob address, its first entry and count, and each entry's token. Work items
  are (group, R output rows), spread over every warp of the grid. A warp reads an expert's weights once for all its
  entries and loads the next step's weights and activations before decoding the current step.
- **fp16 arithmetic.** sm_60 has no `dp4a` but full-rate `half2` FMA, so dot products run in fp16, as in llama.cpp's
  GP100 path: each 32-value block of activations goes to fp16 once, scaled by a power of two so that its largest
  value stays below 2^15, with the exact fp32 block sum kept; products accumulate in fp16 over 8-16 terms and are
  flushed to fp32 with the block scale. The activations keep about 11 bits where the main card's q8_1 path keeps 8, so
  the P100's rows differ from the main card's in the last bits ([quality](#quality)).

### NVFP4 on Pascal

NVFP4 stores 4-bit E2M1 codes with a UE4M3 scale per 16-value sub-block and an fp32 scale per matrix.

- **Codes straight to fp16 bits.** A code `c` (bit 3 the sign, bits 0-2 the magnitude) placed as
  `(c & 7) << 9 | (c & 8) << 12` is exactly the doubled E2M1 value times 2^-15: e = 0 gives an fp16 subnormal, e >= 1
  a normal. No lookup table. Code words are read as aligned 32-bit loads [C:658d01d], [C:574c8c1].
- **No byte permutes.** The kernel stores the consumer's fp16 activations with bits 0 and 1 of the index within each
  group of 4 swapped, so a code word's natural pairs (bytes 0/2 and 1/3) meet consecutive activations [C:574c8c1].
- **Branchless scales.** The UE4M3 scale's 7 bits are placed at fp32 bit 20 and multiplied by 2^(119+K); that is exact
  for normal and subnormal scales (the subnormals land on fp32 denormals), checked bit-exact on all 65,536 inputs
  [R]. The kernel therefore must keep fp32 denormals: no `--use_fast_math`, no `-ftz`.
- **One decode per group**, with only that group's entries (one entry: half the FMAs and loads) [C:658d01d].
- The blob's fp32 tail scale is applied on each matrix's output [C:4ed97b0].

The cubin also has layer kernels for IQ3_XXS, IQ3_S, IQ2_S and IQ4_XS gate/up with IQ4_NL or Q2_0 down (codebooks in
shared memory). Only the NVFP4 pack was benchmarked; the other pairs were not measured.

### Inputs from mapped host memory, results as fp16 rows

- **Inputs.** The kernel reads the window's activations (fp32) and the plan (groups and output rows) straight from
  pinned, mapped host memory over the link [C:2f71e84]. An explicit copy would queue behind the tier's multi-MB expert
  swaps on the same direction of its narrow link. The activations are copied on the host into the tier's own pinned
  buffer first, because the verifier's rows are pinned in the main card's context only.
- **The plan crosses the link once per launch.** Every thread of the grid copies at most one plan record from host
  memory into a device buffer (`LayerArgs::plan`); after the first grid barrier each block loads it from there into
  shared memory. Before 11f2e5b every one of the 112 blocks read the plan over the link, which cost ~80 µs per launch
  at 12 entries and ~160 µs at 35 [R]. The change took the engine's kernel from 259 to 157 µs per launch and decode
  from 110/196/103 to 122/201/122 t/s [C:11f2e5b], [R].
- **Result rows as fp16.** Each result row crosses the link as fp16, scaled by a power of two that puts its largest
  value in [2^14, 2^15), with the per-row scale written beside it [C:024394e]. The host expands the rows into fp32
  with AVX2/F16C (`halves_to_floats`, `src/core/peer_experts.cpp`). That halves the link bytes; fp32 rows measured no
  better (KLD 0.0157 against 0.0150 for fp16 [C:f14e879], [R]).

### Swap rounds on the P100

The P100 follows the conversation like the main card's cache, with rules for its narrow link:

- **Two swaps per round** by default (`kPascalPeerSwaps`; `--peer-adapt-swaps` overrides it). The sweep in
  `src/program/serve/state.hpp`, measured when that default was set (5938a3a, before the kernel work that followed),
  conv/struct/prose t/s: 2 swaps 101/193/98, 4: 98/193/97, 8: 89/185/84, 16: 84/170/83 [C:5938a3a].
- **One round in flight, one stage per decode window** (`Runtime::peer_step`): once the evictions have landed the
  refills go out; once the refills have landed the experts are resident. Stages advance at fixed windows, so runs
  stay reproducible [C:5938a3a].
- **Stepped once per window** (98f6932). The decode loop stepped the P100's round twice per window: `peer_step()`
  issued the refills and `apply_pending()` then waited for those copies (~5.5 MB over the x4 link, ~2 ms with the main
  card idle). Now only `peer_step()` steps it. Time per window: conv -5.0%, struct -4.4%, prose -1.5% [C:98f6932], [R].
- **Promotions.** An expert on the P100 moves to the main card once it has 10 more routings than the main card's
  eviction candidate; a margin of 0 filled the link with promotions [C:5938a3a]. A promoted expert's P100 slot keeps
  its now duplicate copy instead of taking the expert the main card gave up, because that copy would delay the
  layers' rows behind it.
- **Windows: pageable RAM through pinned slots** (6e4cc31). Under WDDM the driver page-locks only part of RAM, so the
  resident expert copy stays pageable and each swap copy held the decode loop while the driver staged it. Swap copies
  with a pageable RAM side now go through small pinned slots (`include/basalt/core/swap_staging.hpp`); the P100's
  evictions, refills and promotion refills use slots of their own, and the main card's copies run on a copier thread
  beside the decode window. Windows decode went from 91/172/89 to 116/187/112 t/s (the f14e879 and 6e4cc31 rows of
  the [Windows](#windows) table).

### Prompt processing

The Pascal kernels are decode-only. With a Pascal tier the engine logs
`pascal tier: decode windows only (no prompt share, no prompt-time split)` and keeps the prompt chunks' peer share,
the prompt-time layer split and the device-planned MoE tail on the main card.

The P100 still helps prompts in one way. With `--peer-exclusive`, the experts only the P100 holds have no RAM copy, so
a prompt chunk has to get them from somewhere. A fixed share, 112 of 256 picked by a hash of (layer, expert), is
copied out of the P100 over its link; the rest are read from the pack file. The link (~2.4 GB/s behind the chipset)
and the SSD then stream side by side [C:dbae67b], [C:9bbf8dd]. The sweep is in `src/prefill/layout.hpp`
(prefill t/s at 16K / 64K tokens):

| share of 256 | 0 | 64 | 96 | 112 | 128 | 154 | 179 | 204 | all over the link |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16K | 3232 | 3729 | 4021 | 4121 | 4065 | 3795 | 3493 | 3223 | 1569 |
| 64K | 3296 | 3783 | 4055 | 4119 | 4085 | 3783 | 3483 | 3217 | |

### The auto plan: peer-exclusive with a Pascal tier (b1049ff)

`peer-exclusive` keeps the tier card's experts only on that card, with no copy in RAM, which frees RAM for the
next-ranked experts. Upstream's automatic rule turns it on only when the full expert set does not fit in free RAM but
the set without the tier's share does. A Blackwell tier needs those RAM copies for the prompt-time split; a Pascal
tier never splits a prompt, so its RAM copies only crowd out other experts. With a peer the prompt path does not split
over, commit b1049ff turns peer-exclusive on whenever the full set does not fit (`src/program/auto_plan.cpp`, test in
`tests/program/auto_plan_test.cpp`). Blackwell tiers keep the old rule. At 61 GB free the choice did not change
(exclusive was already on); at 32 GB it decides whether the P100 helps at all ([table](#32-gb-of-ram-linux)) [R].

In this branch b1049ff is cherry-picked onto the Windows work (as commit `375135f`); the original is on branch
`p100-peer`.

## Driver setup: Linux

NVIDIA supports Blackwell only with its open kernel modules, and the open modules do not support Pascal. Pascal is
supported by the proprietary (closed) kernel modules, and R580 is the newest branch that still binds it (R590 drivers
such as 591.59 have no Pascal entries [K]). The proprietary R580 resource manager refuses a GB20x card unless an
undocumented registry key is set: its only reference to `Rm4598628` compares the value with `0xbf5ae9c6` and
otherwise prints "requires use of the NVIDIA open kernel modules" [W], [R].

The tested setup:

- Ubuntu 26.04 LTS, kernel 7.0.0-31-generic, NVIDIA's CUDA repository, driver 580.196.07 as the proprietary DKMS
  package (`nvidia-dkms-580`, not `nvidia-dkms-580-open`; `modinfo -F license nvidia` prints `NVIDIA`).
- `/etc/modprobe.d/90-nvidia-p100-rtx50.conf` (any name works), exactly this line [R]:

  ```
  options nvidia NVreg_EnableResizableBar=1 NVreg_RegistryDwords="RMDisableFeatureDisablement=1;EnableResizableBar=1;Rm4598628=0xbf5ae9c6"
  ```

  then `sudo update-initramfs -u` and a reboot. `Rm4598628=0xbf5ae9c6` is the key that lets the closed driver bring
  up the RTX 5080. The other two keys were carried over from the P2P setup; whether `Rm4598628` alone is enough was
  never tested in isolation. The static analysis found `RMDisableFeatureDisablement` unrelated to the GB20x check [W].
- Stock 580.196.07 with exactly that line ran both GPUs [R], [S:35fc573e]. Stock 580.178.04 binaries also ran both
  GPUs, with the larger key set of the P2P configuration [R]. The key is undocumented and could stop working with any
  driver update.

**Basalt does not use P2P.** It never enables peer access: the decode path goes through mapped host memory, and its
few `cudaMemcpyPeer` copies (promotions from the P100, the prompt share) are staged through RAM by CUDA [R],
[S:ba873ef6]. Measured handoff round trips: 12.1 µs with the CPU launching and waiting per hop, 4.3 µs between
GPU-side persistent kernels polling host-mapped flags, 2.8 µs with P2P flags; over ~48 round trips per token,
taking the CPU out would save about 0.4 ms of ~21 ms (~2%) and P2P on top about 0.07 ms (0.3%) [R]. P2P between the
two cards needs a patched driver (the closed resource manager, `libcuda` and UVM) and more registry keys [R]; that
work is documented, as an optional extra that Basalt does not need, in
[P2P_P100_RTX50.md](https://github.com/dr3murr/open-gpu-kernel-modules/blob/p100-rtx50-p2p-580.173.02/P2P_P100_RTX50.md)
on the open-gpu-kernel-modules fork.

## Driver setup: Windows

Summary of [contrib/windows-p100/README.md](contrib/windows-p100/README.md), which has the scripts and the steps:

- **One package for both cards.** Windows has one `nvlddmkm` service, so both GPUs must bind to the same package. The
  Data Center 582.99 package (R580) lists the P100 and the RTX PRO 4500/4000 Blackwell (the same GB203 die as the RTX
  5080) but not the RTX 5080; GeForce R590 packages list the 5080 but no Pascal card. The Data Center and GeForce
  builds of one driver version ship the same `nvlddmkm.sys` code; they differ in the INF, the user-mode extras and
  the signature [W].
- **INF edit.** Two lines in `nv_dispwi.inf`: the RTX 5080 (`PCI\VEN_10DE&DEV_2C02`) mapped to the RTX PRO 4500/4000
  Blackwell's install section (`Section094`), and a P100 line without a subsystem ID (`PCI\VEN_10DE&DEV_15F8`) mapped
  to the stock P100 section (`Section026`), plus their names. `contrib/windows-p100/scripts/patch_inf.ps1` applies it
  byte-exactly to your own downloaded copy.
- **Catalog re-sign, as NVCleanstall does.** The edit invalidates the package's catalog, so the kit rebuilds it, signs
  it with a fresh self-signed certificate, deletes the private key and trusts the certificate in LocalMachine Root and
  TrustedPublisher. The driver binaries keep NVIDIA's embedded signatures, so no test-signing is needed and Secure
  Boot stays on. The README explains what trusting such a certificate means and how `rollback.ps1` removes it.
- **Driver models.** The RTX 5080 runs in WDDM (display), the P100 in MCDM (compute only: CUDA and WSL see it, DXGI does
  not). TCC, the INF's default for the P100, fails to start (Code 10): the INF sets `DmaRemappingCompatible = 3`,
  which leaves IOMMU domain setup to the driver stack, and dxgkrnl does that only for WDDM and MCDM adapters, never for
  a TCC one ([DIAGNOSIS.md](contrib/windows-p100/DIAGNOSIS.md), traced with DTrace and the kernel in IDA).
- **Limits.** Windows caps page-locked memory (~15.4 GiB per allocation and 28 GiB in total were measured), and the P100
  has 14.2 GiB usable under MCDM against 15.0 GiB on Linux [K].

## Building

Basalt itself needs CUDA 13 for sm_120; the P100's kernels need a CUDA 12.x nvcc (CUDA 12.9 was used). Without
`BASALT_PASCAL_NVCC` the build has no Pascal kernels, and a run that lists a Pascal card stops with
`pascal tier: this build has no Pascal kernels (configure with -DBASALT_PASCAL_NVCC=<a CUDA 12 nvcc>)`.

Linux (the configuration of the measured builds: GCC 15.2 as the host compiler, `WERROR` and `BUILD_TESTING` off;
CUDA 12.9's nvcc does not accept GCC 15, hence `BASALT_PASCAL_CCBIN`). These were builds of branch `p100-peer`
(f14e879 + b1049ff); this branch adds 086814a and 6e4cc31, which were first compiled with MSVC. Its own Linux check is
described after the command:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
    -DBASALT_PASCAL_NVCC=/usr/local/cuda-12.9/bin/nvcc \
    -DBASALT_PASCAL_CCBIN=/usr/bin/g++-14
cmake --build build -j
```

**Linux check of this branch (2026-11-08): compiled, CPU tests only.** A fresh clone of commit 3320e3e was built in
WSL2 (Ubuntu 24.04) with GCC 13.3, CUDA 13.4 for sm_120 and CUDA 12.9's nvcc for the cubin
(`-DBASALT_PASCAL_NVCC=<12.9 nvcc> -DBASALT_PASCAL_CCBIN=/usr/bin/g++-13`), with the options of upstream's Linux CI job
(`-G Ninja -DWERROR=ON -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=120`, Python 3.12 with `requirements-test.txt`,
which pulls in `requirements.txt`, and pytest):

- All 630 build steps succeeded. No compiler warnings in the project's sources; CUDA 12.9's nvcc printed two for the
  cubin (#177-D: the `NS` members of `Fmt<kIQ4_NL>` and `Fmt<kQ2_0>` are never referenced), which `WERROR` does not
  cover.
- `ctest -L cpu` with every GPU hidden (`CUDA_VISIBLE_DEVICES=-1`): 68 of 68 passed, including `auto_plan_test` and
  the two AVX-512 tests.
- CI's other checks passed: every test has a `cpu` or `gpu` label, `tools/check_isa_isolation.py`, and the lint job
  (`tools/check_platform_boundary.sh`, `tools/check_stderr_logging.sh`, and the server, tool, converter and migration
  Python tests).

Not checked on Linux: anything that runs on a GPU (the `gpu`-labelled tests, the Pascal kernels, the benchmarks).
Under WSL the cards run on the Windows driver, not the Linux setup above, and they were in use by other work. GCC 15.2
and CUDA 13.3, the compilers of the measured builds, were not used for this check.

Windows: MSVC v143 (19.44) from a Visual Studio developer shell, Ninja, CUDA 13.3 and CUDA 12.9 (the 12.9 nvcc only
for the cubin); `contrib/windows-p100/scripts/5_install_toolchain.ps1` installs the parts and `6_build_basalt.ps1`
runs:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON ^
    -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe" ^
    -DBASALT_PASCAL_NVCC="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe"
cmake --build build -j
ctest --test-dir build -L cpu -j 8 --output-on-failure
```

On Windows 61 of 63 CPU tests pass; `iq_avx512_test` (MSVC's float results differ from hashes frozen from a GCC
build in the last bits) and `platform_system_test` (`keep_full_speed`, refused when power throttling is off
system-wide) fail on the test machine [K]. A fresh clone of this branch, built on 2026-11-08 with the kit's
`6_build_basalt.ps1` and no Python with numpy at configure time (which leaves three Python-dependent tests
unregistered), passed 58 of 60 CPU tests with the same two failures; `auto_plan_test` passed. Everything lands in
`bin/` (`basalt`, `basalt-bench`, `basalt-probe`, `basalt-vision`).

## Running

List the main card first and the P100 after it:

```sh
bin/basalt serve --config flashnext.json
```

```json
{
  "server": { "model": "/path/to/Qwen3.8-FN-NVFP4-DQ6_K.basalt", "gpu": "0,1", "host": "127.0.0.1", "port": 8090,
              "min-p": 0.05, "fit-max-tokens": true },
  "engine": { "max-context": 262144, "kv": "int8", "resident-experts": true, "vram-reserve-mib": 700,
              "peer-exclusive": true }
}
```

This is the configuration the measurements served with (the Windows kit writes it in `8_run_basalt.ps1`).
`min-p 0.05` is a deliberate deviation from the model card's 0.0: on the test machine it removed early stops after a
non-canonical token ("`nt` + `dl`" instead of "`nt` + `dll`", then end-of-text) [R].

- `gpu "0,1"` / `--gpu 0,1`: the main card must be sm_120; a later card may be sm_6x (`gpu_tier_supported`,
  `src/program/gpu_select.cpp`). `CUDA_VISIBLE_DEVICES` reorders as upstream documents.
- `peer-exclusive`: with b1049ff the engine picks it by itself whenever the experts do not all fit in RAM; setting it
  explicitly is what was measured.
- `peer-adapt-swaps`: the P100's swaps per round, 2 by default with a Pascal tier.
- `basalt-bench --model PACK --gpu 0,1 [--peer-exclusive] -t conv,struct,prose -r 2 --seed 1 -- --decode-timing`
  measures decode; `--profile` adds the per-stage GPU breakdown.

## Results

### Linux, by commit

`basalt-bench -t conv,struct,prose -r 2 --seed 1`, decode t/s, RTX 5080 + P100 [R]:

| build | conv | struct | prose | P100 kernel per launch |
|---|---:|---:|---:|---:|
| 5080 only (`--gpu 0`) | 82 | 83 | 80 | - |
| 658d01d | 104 | 194 | 102 | - |
| 024394e (fp16 result rows) | 101 | 195 | 103 | 261 µs |
| 574c8c1 (NVFP4 decode rework) | 110 | 196 | 103 | 259 µs |
| 11f2e5b (plan copied once per launch) | 122 | 201 | 122 | 157 µs |

98f6932 against 11f2e5b, time per decode window, alternating A/B, 2 runs each [R]:

| ms per window | 11f2e5b | 98f6932 | change |
|---|---:|---:|---:|
| conv | 22.59 | 21.46 | -5.0% |
| struct | 43.58 | 41.65 | -4.4% |
| prose | 18.50 | 18.23 | -1.5% |

f14e879 (measurement code removed) decodes within 1% of 98f6932 (+0.2%, -0.1%, +0.6%) [R], and its quality rows are
bit-identical to 98f6932's on 10 wikitext chunks [C:f14e879]. Run-to-run noise is about ±3 t/s, so single runs do
not resolve changes under ~5% [R].

In daily use through the server (164 replies of 100 or more tokens, 11f2e5b) the median was 112 t/s, range 86-161.
Draft acceptance explains most of the spread (about +1 t/s per point of acceptance); context depth costs about
-6 t/s per 100k tokens [R].

### Windows

Windows 11, `basalt-bench -t conv,struct,prose -r 2 --seed 1 --gpu 0,1` unless noted, decode t/s [K]. "f14e879" is
f14e879 plus 086814a, a compile fix for MSVC.

| setup | conv | struct | prose |
|---|---:|---:|---:|
| f14e879, automatic plan (P100 mirrors RAM), 48 GB free | 66.6 (run 1), then stopped by the RAM guard | | |
| f14e879, `--peer-exclusive`, 48 GB free (2.9 GiB of experts on the SSD, swaps deferred) | 108 | 89 | 101 |
| f14e879, `--peer-exclusive --resident-budget-gib 32`, 48 GB free | 68 ± 13 | 151 ± 8 | 71 |
| f14e879, `--peer-exclusive`, 58 GB free (apps closed; nothing on the SSD) | 91 ± 3 | 172 ± 0 | 89 ± 3 |
| 6e4cc31, `--peer-exclusive`, 55 GB free (swaps through pinned slots; mean of 3 runs) | 116 | 187 | 112 |
| f14e879, RTX 5080 alone (`--gpu 0`), 58 GB free (8.1 GiB of experts on the SSD, swaps deferred) | 70 ± 0 | 66 ± 1 | 70 ± 1 |

The three 6e4cc31 runs [L]: conv 115/117/116, struct 187/187/187, prose 109/114/113 t/s; draft acceptance 71-74%
(conv), 98% (struct), 66-68% (prose). Free RAM decides most of the Windows result: close other programs before
starting the server [K].

### 32 GB of RAM (Linux)

A 32 GB machine emulated by holding 31.1 GiB of RAM (no swap; 29.4 GiB available), stock driver 580.196.07, int8 KV,
`basalt-bench -t conv,struct,prose -p 16384 -r 2 --seed 1` [R]:

| 32 GB RAM | experts read from the SSD | conv | struct | prose | prefill 16K t/s |
|---|---:|---:|---:|---:|---:|
| 5080 alone | 59% | 31 | 42 | 30 | 972 |
| + P100, old automatic plan (P100 mirrors RAM) | 60% | 18 | 38 | 19 | 941 |
| + P100, `--peer-exclusive` | 36% | 51 | 80 | 45 | 1081 |
| + P100, automatic plan of b1049ff | 37% | 48 | 78 | 43 | 1085 |

With the old plan the P100 held copies of the hottest RAM experts, added no distinct capacity and left the CPU
reading most of its experts from the SSD (137 against 67 ms of CPU work per window); it was slower than the 5080 alone.
With b1049ff's automatic plan the P100 adds +55% / +86% / +43% decode and +12% prefill over the 5080 alone at 32 GB
(computed from the table) [R].

### Quality

No BF16 reference was available, so the P100 path was compared with the 5080-only path using Basalt's quality gate
(`tools/ppl_gate.py`): wikitext-2 test set, 40 chunks of 512 tokens, 10,200 rows teacher-forced through the decode
path (verify width T=3, adaptive swaps off, `--expert-cache 3400` pinned), against a reference written from a
5080-only run by a temporary hook (not part of this branch) [R]:

| run, against the 5080-only reference | KLD mean | KLD p99 | same top-1 | PPL |
|---|---:|---:|---:|---:|
| 5080-only rerun, same pinned configuration | 0.0044 | 0.072 | 98.50% | 2.6485 |
| 5080-only, verify width T=5 | 0.0094 | 0.130 | 97.13% | 2.6433 |
| 5080-only, 3300 cache slots instead of 3400 | 0.0145 | 0.186 | 96.42% | 2.6412 |
| **5080 + P100 (fp16 result rows)** | **0.0150** | 0.196 | 96.19% | 2.6455 |
| 5080 + P100, fp32 result rows | 0.0157 | 0.211 | 96.23% | 2.6464 |
| 5080 + P100, T=5 | 0.0155 | 0.221 | 96.23% | 2.6418 |

Reference PPL 2.6426 ± 0.051. The P100 moves ~52% of routed experts from the CPU to its kernel, which is a placement
change, and it scores like one: the P100 path deviates from the 5080-only path about as much as a 100-slot change of
the expert cache. PPL is unchanged within its standard error. Not measured: quality against BF16, and quality on
Windows [R].

## Where the time goes

One conv window on the 5080 (`--profile`, which disables PDL, so the times read a little high) [R]:

| activity | ms | share |
|---|---:|---:|
| attention, projections, router, shared expert | ~8.1 | 33% |
| experts cached in VRAM | 2.4 | 10% |
| combine, head, other | 1.8 | 7% |
| draft model | 1.0 | 4% |
| waiting for host experts (CPU + P100) | 9.6 | 39% |
| idle before verify (before 98f6932) | 3.4 | 14% |

- The P100's share of the host side, i.e. the time the CPU waits for it after its own experts: ~3.0 ms per window on
  conv, ~2.7 on prose, ~1.2 on struct [R]. In a separate warm 1,880-token decode (19.7 ms per window,
  `--decode-timing`) the wait was 1.9 ms per window, and the routing plan including the P100 launch 0.5 ms [R].
- Without the P100 the CPU computes about 8x more experts per layer (conv: ~20 per layer and window instead of ~2.4)
  [R].
- The P100 kernel at ~10 rows per launch (kernel harness, not part of this branch): gate/up ~68 µs, down ~54 µs,
  result rows to the host ~27 µs, intermediate prep ~15 µs, input prep ~10 µs, launch ~10 µs; the grid barriers cost
  1-4 µs [R].
- Windows after 6e4cc31, about 2 ms per conv window behind Linux: the copier thread's memcpy (~10 ms per window on
  one core, beside the window), the memcpy of the P100's refills, no PCIe reads of experts from RAM (the GPU cannot
  read pageable memory), and 0.8 GiB less room on the P100 under MCDM [K].

## Ceilings

Estimates from the measurements above [R]:

- **P100 wait at zero:** ~139 conv / ~201 struct / ~140 prose t/s; past that the CPU's own experts are the limit.
  11f2e5b was at ~87% of this on conv and prose; struct was already there.
- **P100 hardware floor:** ~95 µs per layer (~48 µs HBM reads, ~37 µs link, ~10 µs launch), against ~170 µs measured.
  Reaching ~125 µs would already make the CPU the limit. The cheap kernel ideas are used up; what is left is the
  ~65 µs fixed cost per launch.
- **Moving experts to the CPU does not pay:** ~50 µs per expert on the CPU against ~11.5 µs on the P100, and the P100
  is full (15.0 of 15.0 GiB).
- **Beyond:** with both the P100 wait and the CPU experts free, conv would top out around ~190 t/s; that needs more
  VRAM.
- **Slot and clock** (assessment, not measured): a CPU-attached x8 slot would double the P100's link bandwidth, which
  helps swaps and the prompt share but not the decode critical path, which is bound by the P100's compute. Raising
  the P100's application clock to 1328 MHz (`nvidia-smi -ac 715,1328`, stock and reversible) was estimated at
  ~1% at most [S:3bd91e10].

## Tried and dropped

All measured on Linux [R]; the patches are not part of this branch.

| attempt | result |
|---|---|
| persistent P100 kernel (doorbell instead of a launch) | works; 259.5 against 249 µs: slower |
| lane-per-row down phase (no idle lanes, no reductions) | correct, no gain; larger steps 2x slower |
| dynamic work scheduling (atomic counter) | harness -3 to -16 µs, no gain in the engine A/B |
| tile sizes `BP_RG`/`BP_RD`/`BP_MINB` | the current 2/4/2 is best overall; RG=1 helps small launches ~9 µs, hurts big ones ~10 µs |
| smaller main-card swap rounds (`--adapt-swaps 48/24`) | no change within noise |
| no adaptive swaps (`--adapt-swaps 0`) | removes the swap stall, but struct 188 -> 100 t/s |
| pipelined windows in the expert tier (a two-stage split on the 5080, window B's first half beside A's second) | runs after two fixes, no net gain: two whole-card kernels cannot share the card, the pair rarely launches on conv and prose, and on struct the overlap saves ~7% per window while pair windows carry ~10% fewer tokens. Dropped |

Two lessons from the persistent kernel on sm_60: a value re-read after a `.cv` spin loop on host memory can come back
stale, and volatile reads of system memory can be served from L2 for milliseconds. On the 5080, polling host-mapped
flags worked only with `ld.acquire.sys`; `ld.global.cv` never saw the update [R].

## Caveats

- One machine, one model, one pack. Only `Qwen3.8-FN-NVFP4-DQ6_K` was measured; the IQ/Q2_0 kernel pairs in the
  cubin were not. Only the P100 was tested as the Pascal card. Other sm_6x cards (P40, P4, GTX 10xx) pass the
  card check but are untested; their chips (GP102 and smaller) run fp16 arithmetic, which these kernels are built on,
  at a small fraction of the P100's rate, and they have less memory bandwidth than the P100's HBM2.
- The driver setups are modified (see the top). The Linux key is undocumented; the Windows package carries a catalog
  signed by your own certificate. A driver update can break either.
- Windows Update or an NVIDIA installer can replace the edited package; the kit blocks driver updates from Windows
  Update.
- The RTX 5080 drives the display, so it is subject to TDR on Windows (2 s by default) [K].
- The headline Linux numbers were taken with the P2P-patched 580.173.02 loaded, not with the stock 580.196.07 setup
  this document recommends; the stock setup was checked with a served request (124.9 t/s) and the 32 GB runs
  [S:ba873ef6], [R].
- On Linux the 5080 crashed twice on 2026-11-03 on the proprietary 580.173.02 driver: `Xid 62` (PMU halt), then
  `Xid 154` (GPU reset required), both while it sat idle at a bare text console. The likely cause, not proven, is the
  driver's idle-display power saving (DIFR). A locally rebuilt `nvidia-modeset` with DIFR turned off (not published)
  then ran 19.5 hours without an Xid, and none has been seen since the machine runs X, which is not proof [R].
- Run-to-run noise is about ±3 t/s; free RAM changes results a lot (Windows table).
- The fp16 kernels round differently from the main card's kernels; results are reproducible run to run only with the
  expert cache pinned (`--expert-cache N`) and the same expert placement [R].
- Upstream Basalt has moved on since `9595ae0`; this branch has not been rebased or reviewed upstream.
- On Linux this branch was only compiled and CPU-tested (2026-11-08, GCC 13.3 in WSL2; [Building](#building)); it was
  not run on the GPUs under Linux. The Linux numbers come from builds of `p100-peer`.

## Reproducing

Linux build: [Building](#building). Benchmarks (stop any running server first):

```sh
./bin/basalt-bench --model /path/to/Qwen3.8-FN-NVFP4-DQ6_K.basalt --gpu 0,1 -t conv,struct,prose -r 2 --seed 1 \
    --logs DIR -- --decode-timing [--profile]
./bin/basalt-bench --model /path/to/Qwen3.8-FN-NVFP4-DQ6_K.basalt --gpu 0 -t conv,struct,prose -r 2 --seed 1   # 5080 alone
```

Windows (from the repository root, after steps 1-7 of the kit; `basalt-bench.exe` loads `cublas64_13.dll` from
CUDA 13.3):

```bat
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;%PATH%
bin\basalt-bench.exe --model <dir>\Qwen3.8-FN-NVFP4-DQ6_K.basalt -t conv,struct,prose -r 2 --seed 1 ^
    --gpu 0,1 --peer-exclusive --logs DIR -- --decode-timing
```

32 GB emulation: hold RAM until ~29.4 GiB is available (no swap), then add `-p 16384` to the bench line [R].

The P100 kernel on its own (no engine): build the cubin by hand with
`nvcc -std=c++17 -O3 -cubin -arch=sm_60 -Iinclude -o pascal_peer_sm60.cubin src/kernels/pascal/peer_experts_sm60.cu`
(CUDA 12.x; add `-ccbin g++-14` where the default GCC is newer). The harness used for the per-phase timings is not
part of this branch.

## Commits

On top of upstream `9595ae0`, oldest first:

| commit | what it does |
|---|---|
| a693ca2 | the Pascal (sm_60) expert tier: embedded CUDA 12 cubin, fused layer kernel, pool path (with temporary instrumentation) |
| 4ed97b0 | NVFP4 experts (`bp_layer_40_40`): E2M1 into fp16 bit patterns, UE4M3 sub-block scales, the blob's fp32 tail scale |
| 2f71e84 | inputs and the plan read from mapped host memory; row and occupancy knobs `BP_RG`/`BP_RD`/`BP_MINB` |
| 5938a3a | promotion margin 10; one round in flight, one stage per window (`peer_step`); 2 swaps per round |
| dbae67b | prefill: half of the tier's exclusive experts over its link, half from the pack file |
| 9bbf8dd | prefill: link share by a hash of (layer, expert), 112/256 |
| 658d01d | one decode per group with only its entries; NVFP4 codes straight to fp16 bits |
| 024394e | result rows over the link as fp16, scaled per row by a power of two |
| 574c8c1 | NVFP4 decode: aligned 32-bit loads, branchless UE4M3 scales, no byte permutes |
| 11f2e5b | the plan crosses the link once per launch: kernel 259 -> 157 µs, decode 110/196/103 -> 122/201/122 |
| 98f6932 | the P100's swap round stepped once per window: ms per window -5.0/-4.4/-1.5% |
| f14e879 | measurement scaffolding removed; result rows always fp16 |
| 086814a | Windows: MSVC-compatible target attribute for `halves_to_floats` |
| 6e4cc31 | Windows: swap copies with pageable RAM through pinned slots: 91/172/89 -> 116/187/112 |
| 375135f | auto plan: peer-exclusive with a Pascal tier whenever the full set does not fit (cherry-pick of b1049ff) |

The other two branches of the fork hold the same work unmerged: `p100-peer` (f14e879 + b1049ff, what the Linux
machine ran) and `windows-build` (f14e879 + 086814a + 6e4cc31). `p100-peer` still has the GCC-only
`__attribute__((target(...)))` in `src/core/peer_experts.cpp` that upstream's `tools/check_platform_boundary.sh`
rejects (on 2026-11-08 it failed on `p100-peer` and passed on this branch); 086814a replaces it in the other two
branches.

## Related work

- llama.cpp with P100 kernels and the same "P100 as expert tier" layout for other MoE models:
  [P100.md](https://github.com/dr3murr/llama.cpp/blob/p100-qr0n/P100.md) (branch `p100-qr0n`).
- Qwen3.8-27B on an RTX 5080 + P100 with llama.cpp:
  [QWEN38_27B.md](https://github.com/dr3murr/llama.cpp/blob/qwen38-27b-p100/QWEN38_27B.md) (branch `qwen38-27b-p100`).
- P2P between the RTX 5080 and the P100 on Linux (not used by Basalt):
  [P2P_P100_RTX50.md](https://github.com/dr3murr/open-gpu-kernel-modules/blob/p100-rtx50-p2p-580.173.02/P2P_P100_RTX50.md).

## Prior art and credits

- **The expert tier** is upstream Basalt's: every card after the main one holds and computes experts, and the start
  line `multi-gpu: ... ms per window: expert tier ..., layer split ...; picked expert tier` with the rule that unlike
  cards always use the expert tier is upstream code ([docs/configuration/gpus.md](docs/configuration/gpus.md),
  [upstream](https://github.com/jesdga95/basalt/blob/main/docs/configuration/gpus.md)). Basalt took it over from
  [Strata](https://github.com/Niko1221/Strata)'s helper caches and peer tier
  ([SECOND_GPU.md](https://github.com/Niko1221/Strata/blob/main/docs/SECOND_GPU.md)), where a Tesla P100 already ran
  as a helper card next to a V100 ([Strata #1079](https://github.com/Niko1221/Strata/issues/1079),
  [#1452](https://github.com/Niko1221/Strata/pull/1452)).
- **NVFP4 on non-Blackwell cards** was done before in software by Strata's decode kernels, benchmarked on a Tesla
  P100 ([NVFP4.md](https://github.com/Niko1221/Strata/blob/main/docs/NVFP4.md)).
- **The fp16 approach** of the Pascal kernel follows the GP100 path of the llama.cpp fork
  ([P100.md](https://github.com/dr3murr/llama.cpp/blob/p100-qr0n/P100.md)), whose mat-vec kernels build on
  [qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp).
- **What this fork adds:** the tier running on a Pascal card inside Basalt's sm_120, CUDA 13 engine (a separate
  CUDA 12.9 cubin, NVFP4 decoded into fp16 math, inputs from mapped host memory), the plan and swap-round changes for
  a narrow link, the peer-exclusive rule for such a tier, the Windows swap staging, and the measurements. Credits for
  the Windows driver setup are in [contrib/windows-p100/README.md](contrib/windows-p100/README.md#credits-and-prior-art).

## Sources

The numbers come from the author's measurement notes and logs, which are not published; the commit messages are in
this repository.

| key | source |
|---|---|
| [R] | Linux measurement report "Basalt on RTX 5080 + Tesla P100", 2026-11-03 to 2026-11-05 (sections: results, kept changes, tried and dropped, where the time goes, ceilings, quality, open caveats, Flash-Next with 32 GB of RAM) |
| [K] | the Windows kit README, numbers table and known risks; published, adapted, as [contrib/windows-p100/README.md](contrib/windows-p100/README.md) |
| [L] | the Windows benchmark logs of 2026-11-05 (`table.md` of three `basalt-bench` runs; build stamp 086814a with the working tree that was committed as 6e4cc31) |
| [W] | static analysis of the Windows 581.80/582.78/582.99/591.59 and Linux 580.196.07 driver binaries and INFs, 2026-11-04 |
| [C:hash] | the commit message of that commit in this repository |
| [S:id] | a message in the agent session log of 2026-11-01 to 2026-11-05, by message id |
