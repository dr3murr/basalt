#Requires -RunAsAdministrator
# Step 3b (run by 3_install_r580.ps1 and 3e_reinstall_eac.ps1): sets the Tesla P100's driver model to MCDM (default)
# or WDDM, plus the Tesla keys it was tested with. Reboot afterwards; the driver reads all of it at start.
# Why: the INF installs the P100 in TCC (AdapterType 2) and sets DmaRemappingCompatible = 3, which leaves DMA
# remapping of every nvlddmkm device to the driver. dxgkrnl does that for WDDM and MCDM adapters; nothing does it for a
# TCC adapter, so a TCC P100 fails to start (Code 10, 0xC000009A) and stays failed until a reboot (DIAGNOSIS.md).
# MCDM is compute-only: CUDA and WSL see the P100, DXGI (games) does not. AdapterType is written directly because
# nvidia-smi cannot reach a P100 that failed to start.
#   Services\nvlddmkm\Parameters\AllowWddmOnTesla = 1; P100 adapter key: AllowWddmOnTesla = 1,
#   GridLicensedFeatures = 7, EnableMsHybrid = 1 (the values TeslaP40_wddm's enable_wddm.bat writes). Tested together
#   with MCDM; whether MCDM needs them is untested.
param([ValidateSet('MCDM', 'WDDM')][string]$DriverModel = 'MCDM')
$ErrorActionPreference = 'Stop'
$AdapterTypes = @{ WDDM = 1; MCDM = 4 }   # the values nvidia-smi -dm 0 / -dm 2 write
$svc = 'HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Parameters'
$cls = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'

$p100 = Get-ChildItem $cls -ErrorAction SilentlyContinue | Where-Object {
    (Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue).DriverDesc -match 'P100'
}
if (-not $p100) { throw 'no P100 adapter key: install the driver package first (step 3)' }
if (@($p100).Count -gt 1) { throw "more than one P100 adapter key: $($p100.PSChildName -join ', ')" }
$key = $p100.PSPath
Write-Host "P100 adapter key: $($p100.PSChildName)"

if (-not (Test-Path $svc)) { New-Item -Path $svc | Out-Null }   # New-Item -Force on an existing key deletes its values
Set-ItemProperty -Path $svc -Name AllowWddmOnTesla     -Type DWord -Value 1
Set-ItemProperty -Path $key -Name AllowWddmOnTesla     -Type DWord -Value 1
Set-ItemProperty -Path $key -Name GridLicensedFeatures -Type DWord -Value 7
Set-ItemProperty -Path $key -Name EnableMsHybrid       -Type DWord -Value 1
Set-ItemProperty -Path $key -Name AdapterType          -Type DWord -Value $AdapterTypes[$DriverModel]

Get-ItemProperty $key | Select-Object DriverDesc, AdapterType, AllowWddmOnTesla, GridLicensedFeatures, EnableMsHybrid | Format-List
Write-Host "P100 driver model: $DriverModel (AdapterType $($AdapterTypes[$DriverModel])). Reboot for it to take effect."
