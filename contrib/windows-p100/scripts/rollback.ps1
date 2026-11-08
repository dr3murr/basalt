#Requires -RunAsAdministrator
# Rollback: removes every installed NVIDIA display package (the self-signed R580 one), removes every certificate this
# kit created (the thumbprints in $StateDir\cert-thumbprint.txt) from LocalMachine Root, TrustedPublisher and My, and
# reinstalls the driver(s) 1_backup.ps1 exported to $StateDir\driver-backup. Reboot afterwards. What it leaves in
# place is listed in the README ("Rollback").
param([string]$StateDir = (Join-Path $env:ProgramData 'p100-win'))
$ErrorActionPreference = 'Continue'
# Windows PowerShell's own module paths (PS7's, inherited when launched from pwsh, break the Cert: drive)
$env:PSModulePath = "$env:USERPROFILE\Documents\WindowsPowerShell\Modules;$env:ProgramFiles\WindowsPowerShell\Modules;$env:SystemRoot\system32\WindowsPowerShell\v1.0\Modules"
$pkgs = Get-WindowsDriver -Online | Where-Object { $_.ProviderName -match 'NVIDIA' -and $_.ClassName -eq 'Display' }
foreach ($p in $pkgs) {
    Write-Host "removing $($p.Driver) $($p.Version)"
    pnputil /delete-driver $p.Driver /uninstall /force
}
. (Join-Path $PSScriptRoot 'signing_cert.ps1')
$thumbFile = Join-Path $StateDir 'cert-thumbprint.txt'
$thumbs = @(Get-Content $thumbFile -ErrorAction SilentlyContinue | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($thumbs.Count -eq 0) {
    Write-Warning "no certificates recorded in ${thumbFile}: check certlm.msc (Trusted Root Certification Authorities, Trusted Publishers) for the subject you signed with"
} else {
    try {
        Remove-CatalogSigningCert $thumbs $thumbFile
        Write-Host "certificates removed: $($thumbs -join ', ')"
    } catch {
        Write-Warning "certificate removal failed ($_); remove $($thumbs -join ', ') from LocalMachine Root, TrustedPublisher and My by hand (certlm.msc)"
    }
}
$bak = Join-Path $StateDir 'driver-backup'
if (-not (Test-Path $bak)) { Write-Warning "no driver backup in ${bak}: install your previous NVIDIA driver with its own installer" }
Get-ChildItem $bak -Recurse -Filter '*.inf' -ErrorAction SilentlyContinue | ForEach-Object { pnputil /add-driver $_.FullName /install }
Write-Host 'Reboot. (Or run your previous NVIDIA installer.) To allow Windows Update drivers again, delete'
Write-Host 'HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\ExcludeWUDriversInQualityUpdate.'
