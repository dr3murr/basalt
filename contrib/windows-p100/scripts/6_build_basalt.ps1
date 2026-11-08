# Step 6: builds Basalt natively with Ninja and runs its CPU tests. CUDA 13.3 builds the engine (as on Linux); CUDA
# 12.9's nvcc builds the P100 tier's sm_60 cubin, which CUDA 13 cannot; MSVC v143 (19.44) is the host compiler for
# both. -Src defaults to the checkout this script sits in; the programs land in $Src\bin whatever -Build is.
# On the test machine 61 of 63 CPU tests pass; iq_avx512_test and platform_system_test fail (README, "Known test
# failures"), so this script stops with "CPU tests failed" after a complete build.
param(
    [string]$Src = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path,
    [string]$Build = '',   # empty: $Src\build
    [string]$Cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3',
    [string]$PascalCuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9',
    [string]$MsvcVersion = '14.44',
    [int]$Jobs = 0,        # parallel compile jobs; 0: Ninja's default (cores + 2). Lower it when RAM is short
    [int]$TestJobs = 8,    # parallel CPU tests
    [switch]$NoTests
)
$ErrorActionPreference = 'Stop'
if (-not $Build) { $Build = Join-Path $Src 'build' }
foreach ($nvcc in "$Cuda\bin\nvcc.exe", "$PascalCuda\bin\nvcc.exe") {
    if (-not (Test-Path $nvcc)) { throw "$nvcc not found: run 5_install_toolchain.ps1" }
}
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64 -vcvars_ver=$MsvcVersion"
$cl = (Get-Command cl.exe).Source
if ($cl -notmatch "\\MSVC\\$([regex]::Escape($MsvcVersion))") { throw "cl.exe is $cl, not MSVC $MsvcVersion" }
$env:CUDA_PATH = $Cuda
$env:PATH = "$Cuda\bin;$env:PATH"

cmake -S $Src -B $Build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON `
    -DCMAKE_CUDA_COMPILER="$($Cuda -replace '\\','/')/bin/nvcc.exe" `
    -DBASALT_PASCAL_NVCC="$($PascalCuda -replace '\\','/')/bin/nvcc.exe"
if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed' }
if ($Jobs -gt 0) { cmake --build $Build -j $Jobs } else { cmake --build $Build -j }
if ($LASTEXITCODE -ne 0) { throw 'build failed' }
if (-not $NoTests) {
    ctest --test-dir $Build -L cpu -j $TestJobs --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw 'CPU tests failed (two known failures on Windows, see the README; the programs are built)' }
}
Write-Host "built: $Src\bin (run 8_run_basalt.ps1)"
