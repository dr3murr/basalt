#Requires -RunAsAdministrator
# Step 1: back up the installed NVIDIA display driver package(s) into $StateDir\driver-backup, keep Windows Update from
# replacing drivers, and make a restore point. Safe to run more than once.
param([string]$StateDir = (Join-Path $env:ProgramData 'p100-win'))
$ErrorActionPreference = 'Stop'
$bak = Join-Path $StateDir 'driver-backup'
$marker = Join-Path $StateDir 'packages.csv'
New-Item -ItemType Directory -Force $bak | Out-Null
Remove-Item $marker -Force -ErrorAction SilentlyContinue   # only written once every export is verified

$pkgs = @(Get-WindowsDriver -Online | Where-Object { $_.ProviderName -match 'NVIDIA' -and $_.ClassName -eq 'Display' })
if ($pkgs.Count -eq 0) { throw 'no installed NVIDIA display driver package found: nothing to back up' }
$pkgs | Select-Object Driver, Version, OriginalFileName | Format-Table -AutoSize | Out-String | Write-Host
foreach ($p in $pkgs) {
    $dest = Join-Path $bak $p.Driver
    New-Item -ItemType Directory -Force $dest | Out-Null
    pnputil /export-driver $p.Driver $dest
    if ($LASTEXITCODE -ne 0) { throw "pnputil /export-driver $($p.Driver) failed (exit $LASTEXITCODE)" }
    $inf = Get-ChildItem $dest -Recurse -Filter '*.inf' -ErrorAction SilentlyContinue
    $sys = Get-ChildItem $dest -Recurse -Filter 'nvlddmkm.sys' -ErrorAction SilentlyContinue
    if (-not $inf -or -not $sys) { throw "backup of $($p.Driver) in $dest has no .inf or no nvlddmkm.sys: not usable" }
    Write-Host "backed up $($p.Driver): $(@(Get-ChildItem $dest -Recurse -File).Count) files"
}
# the marker step 2 checks: written only after every package exported and verified
$pkgs | Select-Object Driver, Version, OriginalFileName | Export-Csv $marker -NoTypeInformation

# no drivers from Windows Update (it would put a WHQL GeForce driver back on the 5080)
$wu = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate'
if (-not (Test-Path $wu)) { New-Item -Path $wu | Out-Null }   # New-Item -Force on an existing key deletes its values
Set-ItemProperty -Path $wu -Name ExcludeWUDriversInQualityUpdate -Type DWord -Value 1
Set-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\DriverSearching' -Name SearchOrderConfig -Type DWord -Value 0

try {
    Checkpoint-Computer -Description 'before p100-win R580 driver' -RestorePointType MODIFY_SETTINGS
    Write-Host 'restore point created'
} catch {
    Write-Warning "no restore point: $($_.Exception.Message) (System Restore may be off; the driver backup in $bak is enough)"
}
Write-Host "backup in $bak. Next: 2_remove_current.ps1"
