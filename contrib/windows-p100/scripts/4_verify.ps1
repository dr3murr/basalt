# Step 4: check both GPUs bound to 582.99 (5080 WDDM, P100 MCDM), their DMA policy, and CUDA on each.
Write-Host '== devices (DMA policy 2 = the driver manages DMA remapping, from DmaRemappingCompatible = 3)'
Get-PnpDevice -Class Display | Where-Object { $_.InstanceId -match 'VEN_10DE' } | ForEach-Object {
    $policy = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_DmaRemappingPolicy -ErrorAction SilentlyContinue).Data
    $status = [uint32](Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_ProblemStatus -ErrorAction SilentlyContinue).Data
    [pscustomobject]@{ Device = $_.FriendlyName; Status = $_.Status; Problem = $_.Problem; ProblemStatus = '0x{0:X8}' -f $status; DmaPolicy = $policy }
} | Format-Table -AutoSize
Write-Host '== nvidia-smi'
nvidia-smi --query-gpu=index,name,driver_version,driver_model.current,driver_model.pending,pci.bus_id --format=csv
Write-Host '== registry (AdapterType: 1 WDDM, 2 TCC, 4 MCDM)'
$cls = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
Get-ChildItem $cls -ErrorAction SilentlyContinue | ForEach-Object {
    $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
    if ($p.DriverDesc -match 'NVIDIA') {
        [pscustomobject]@{ Key = $_.PSChildName; Device = $p.DriverDesc; Version = $p.DriverVersion; AdapterType = $p.AdapterType; AllowWddmOnTesla = $p.AllowWddmOnTesla; GridLicensedFeatures = $p.GridLicensedFeatures; EnableMsHybrid = $p.EnableMsHybrid }
    }
} | Format-Table -AutoSize
$params = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Parameters' -ErrorAction SilentlyContinue
Write-Host "Services\nvlddmkm\Parameters: DmaRemappingCompatible = $($params.DmaRemappingCompatible), AllowWddmOnTesla = $($params.AllowWddmOnTesla)"
Write-Host '== CUDA (allocate, fill and read back 1 GiB on every device)'
if (Get-Command python -ErrorAction SilentlyContinue) { python "$PSScriptRoot\cuda_smoke.py" }
else { Write-Host 'python not on PATH: skipped (cuda_smoke.py needs only Python and the driver)' }
Write-Host '== nvlddmkm events (last 15)'
Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = 'nvlddmkm' } -MaxEvents 15 -ErrorAction SilentlyContinue |
    Select-Object TimeCreated, Id, LevelDisplayName, Message | Format-Table -Wrap
Write-Host 'If the P100 shows Code 10 (0xC000009A), it started in TCC: run 3b_p100_driver_model.ps1 and reboot.'
