#Requires -RunAsAdministrator
# Step 5: the Basalt build toolchain. Visual Studio (2022 or 2026, with the C++ workload) gets the MSVC v143 (19.44)
# toolset: the compiler Basalt documents for Windows, and the newest one CUDA 12.9 accepts. CUDA 12.9 provides the nvcc
# for the P100 tier's sm_60 cubin; CUDA 13.3 builds Basalt itself, as on Linux. Only the components Basalt uses are
# installed, never a driver: the CUDA installers' bundled driver would replace the 582.99 package. 12.9 goes first so
# that CUDA_PATH ends at 13.3. Fails if the display driver is no longer 582.99.
param([string]$Downloads = (Join-Path $env:TEMP 'p100-win-installers'))
$ErrorActionPreference = 'Stop'
$V143Component = 'Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64'
$V143Version = '14.44'
$CudaReleases = @(
    @{ Version = '12.9.1'; Parts = @('nvcc', 'cudart') },
    @{ Version = '13.3.1'; Parts = @('nvcc', 'crt', 'nvvm', 'thrust', 'cudart', 'cublas', 'cublas_dev', 'nvjitlink') }   # thrust = CCCL (cub)
)
$CudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'

function Install-V143Toolset {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'no Visual Studio with the C++ tools found' }
    if (Get-ChildItem "$vs\VC\Tools\MSVC" -Directory -Filter "$V143Version.*" -ErrorAction SilentlyContinue) {
        Write-Host "MSVC $V143Version already installed"; return
    }
    $setup = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\setup.exe"
    $p = Start-Process $setup -ArgumentList 'modify', '--installPath', "`"$vs`"", '--add', $V143Component, '--quiet', '--norestart' -Wait -PassThru
    if ($p.ExitCode -notin 0, 3010) { throw "Visual Studio installer failed ($($p.ExitCode))" }
    if (-not (Get-ChildItem "$vs\VC\Tools\MSVC" -Directory -Filter "$V143Version.*")) { throw "MSVC $V143Version missing after install" }
    Write-Host "MSVC $V143Version installed"
}

function Install-CudaParts([string]$Version, [string[]]$Parts) {
    $short = $Version -replace '\.\d+$', ''
    $nvcc = "$CudaRoot\v$short\bin\nvcc.exe"
    if (Test-Path $nvcc) { Write-Host "CUDA $short already installed"; return }
    New-Item -ItemType Directory -Force $Downloads | Out-Null
    $exe = Join-Path $Downloads "cuda_${Version}_windows_network.exe"
    if (-not (Test-Path $exe)) {
        Invoke-WebRequest -UseBasicParsing "https://developer.download.nvidia.com/compute/cuda/$Version/network_installers/cuda_${Version}_windows_network.exe" -OutFile $exe
    }
    $installerArgs = @('-s') + ($Parts | ForEach-Object { "${_}_$short" }) + '-n'
    $p = Start-Process $exe -ArgumentList $installerArgs -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "CUDA $Version installer failed ($($p.ExitCode))" }
    if (-not (Test-Path $nvcc)) { throw "nvcc missing after the CUDA $Version install" }
    & $nvcc --version | Select-Object -Last 2
}

Install-V143Toolset
foreach ($r in $CudaReleases) { Install-CudaParts $r.Version $r.Parts }
$driver = (nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
if ($driver -ne '582.99') { throw "the display driver changed to ${driver}: reinstall with 3e_reinstall_eac.ps1" }
Write-Host "toolchain ready; driver still $driver"
