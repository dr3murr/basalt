#Requires -RunAsAdministrator
# Step 3: rebuild the catalog of your patched 582.99 package (patch_inf.ps1), sign it with a fresh local certificate
# (signing_cert.ps1), delete that certificate's private key, install for both GPUs, then set the P100's driver model
# (3b). The driver binaries are unchanged and keep NVIDIA's embedded signatures: no test-signing, Secure Boot stays
# on. The catalog covers every file of the package (Inf2Cat, from the Windows Driver Kit). 3e_reinstall_eac.ps1 is
# the alternative with an INF-only catalog (Windows SDK only); the test machine ran that one when it was measured.
param(
    [Parameter(Mandatory = $true)][string]$DriverDir,   # the extracted package's Display.Driver folder
    [string]$StateDir = (Join-Path $env:ProgramData 'p100-win'),
    [string]$Inf2Cat = '',    # Inf2Cat.exe; empty: the newest under Windows Kits\10\bin
    [string]$Signtool = '',   # signtool.exe; empty: the newest under Windows Kits\10\bin
    [string]$Subject = ''     # the certificate's subject; empty: signing_cert.ps1's default
)
$ErrorActionPreference = 'Stop'
# launched from PowerShell 7, Windows PowerShell inherits PS7 module paths and loads PS7's incompatible
# Microsoft.PowerShell.Security (no Cert: drive, no certificate cmdlets): use 5.1's own module paths
$env:PSModulePath = "$env:USERPROFILE\Documents\WindowsPowerShell\Modules;$env:ProgramFiles\WindowsPowerShell\Modules;$env:SystemRoot\system32\WindowsPowerShell\v1.0\Modules"
. (Join-Path $PSScriptRoot 'signing_cert.ps1')
$dir = (Resolve-Path $DriverDir).Path
$thumbFile = Join-Path $StateDir 'cert-thumbprint.txt'
if (-not (Test-PatchedInf $dir)) { throw "$dir\nv_dispwi.inf has no RTX 5080 line: run patch_inf.ps1 -DriverDir '$dir' first" }
$Inf2Cat = Find-KitTool 'Inf2Cat.exe' $Inf2Cat
$Signtool = Find-KitTool 'signtool.exe' $Signtool

Write-Host '== 1. catalog (Inf2Cat hashes the whole package, a few minutes)'
Remove-Item "$dir\nv_disp.cat" -Force -ErrorAction SilentlyContinue
& $Inf2Cat /driver:"$dir" /os:10_X64 /uselocaltime
if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$dir\nv_disp.cat")) { throw "Inf2Cat failed ($LASTEXITCODE)" }

Write-Host '== 2. local signing certificate, trusted in LocalMachine Root + TrustedPublisher'
$cert = New-CatalogSigningCert $thumbFile $Subject

Write-Host '== 3. sign the catalog, then delete the private key'
& $Signtool sign /fd sha256 /sm /s My /sha1 $cert.Thumbprint "$dir\nv_disp.cat"
if ($LASTEXITCODE -ne 0) { throw "signtool failed ($LASTEXITCODE)" }
Remove-Item "Cert:\LocalMachine\My\$($cert.Thumbprint)" -DeleteKey
& $Signtool verify /pa /v "$dir\nv_disp.cat" | Select-Object -Last 3
Write-Host "signed by $($cert.Subject) ($($cert.Thumbprint), recorded in $thumbFile); private key deleted"

Write-Host '== 4. install for both GPUs'
pnputil /add-driver "$dir\nv_dispwi.inf" /install
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3010) { throw "pnputil /add-driver failed (exit $LASTEXITCODE); rollback.ps1 reinstalls the backup" }

Write-Host '== 5. P100 driver model (the INF installs it in TCC, which cannot start under its DMA policy)'
& "$PSScriptRoot\3b_p100_driver_model.ps1"
Write-Host 'Reboot now, then run 4_verify.ps1'
