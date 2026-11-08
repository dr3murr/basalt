#Requires -RunAsAdministrator
# Step 2: remove the installed NVIDIA display driver packages. R590 and later (e.g. GeForce 591.59) have no Pascal
# support and, being WHQL-signed, would outrank the self-signed R580 package for the 5080. The screen falls back to
# Microsoft Basic Display until step 3. Refuses to remove anything unless 1_backup.ps1 left a usable backup of every
# package it would remove.
param([string]$StateDir = (Join-Path $env:ProgramData 'p100-win'))
$ErrorActionPreference = 'Stop'
$bak = Join-Path $StateDir 'driver-backup'
$marker = Join-Path $StateDir 'packages.csv'
if (-not (Test-Path $marker)) { throw "no verified backup ($marker missing): run 1_backup.ps1 first" }
$backedUp = @(Import-Csv $marker | ForEach-Object { $_.Driver })

$pkgs = @(Get-WindowsDriver -Online | Where-Object { $_.ProviderName -match 'NVIDIA' -and $_.ClassName -eq 'Display' })
foreach ($p in $pkgs) {   # check every package before removing any
    $dest = Join-Path $bak $p.Driver
    $ok = ($backedUp -contains $p.Driver) -and
          (Get-ChildItem $dest -Recurse -Filter '*.inf' -ErrorAction SilentlyContinue) -and
          (Get-ChildItem $dest -Recurse -Filter 'nvlddmkm.sys' -ErrorAction SilentlyContinue)
    if (-not $ok) { throw "no usable backup of $($p.Driver) ($($p.Version)) in ${dest}: run 1_backup.ps1 again; nothing removed" }
}
foreach ($p in $pkgs) {
    Write-Host "removing $($p.Driver) $($p.Version) $($p.OriginalFileName)"
    pnputil /delete-driver $p.Driver /uninstall /force
    if ($LASTEXITCODE -ne 0) { throw "pnputil /delete-driver $($p.Driver) failed (exit $LASTEXITCODE)" }
}
Write-Host 'Reboot now, then run patch_inf.ps1 (if not done yet) and 3_install_r580.ps1 or 3e_reinstall_eac.ps1'
