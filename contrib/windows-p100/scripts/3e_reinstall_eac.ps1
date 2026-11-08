#Requires -RunAsAdministrator
# Step 3e (alternative to step 3, or run after it): an anti-cheat friendly package, like NVCleanstall's EAC mode.
# The catalog lists only the INFs, so the driver binaries are validated by their own embedded NVIDIA/Microsoft
# signatures and no driver file chains to the local certificate. A fresh certificate signs it (signing_cert.ps1) and
# its private key is deleted right after. Removes any installed NVIDIA display package, installs this one, then
# removes the certificates earlier runs of 3 or 3e created. Windows asks "can't verify the publisher" once;
# -AutoAccept presses "Install this driver software anyway" for you. Needs makecat and signtool (Windows SDK).
# This is the catalog the test machine ran when the numbers in the README were measured.
param(
    [Parameter(Mandatory = $true)][string]$DriverDir,   # the extracted package's Display.Driver folder
    [string]$StateDir = (Join-Path $env:ProgramData 'p100-win'),
    [string]$Subject = '',   # the certificate's subject; empty: signing_cert.ps1's default
    [switch]$AutoAccept
)
$ErrorActionPreference = 'Stop'
# Windows PowerShell's own module paths (PS7's, inherited when launched from pwsh, break the Cert: drive)
$env:PSModulePath = "$env:ProgramFiles\WindowsPowerShell\Modules;$env:SystemRoot\system32\WindowsPowerShell\v1.0\Modules"
. (Join-Path $PSScriptRoot 'signing_cert.ps1')
$dir = (Resolve-Path $DriverDir).Path
if (-not (Test-PatchedInf $dir)) { throw "$dir\nv_dispwi.inf has no RTX 5080 line: run patch_inf.ps1 -DriverDir '$dir' first" }
$makecat = Find-KitTool 'makecat.exe' ''
$signtool = Find-KitTool 'signtool.exe' ''
New-Item -ItemType Directory -Force $StateDir | Out-Null
$log = Join-Path $StateDir '3e_pnputil.log'
$thumbFile = Join-Path $StateDir 'cert-thumbprint.txt'

function New-InfOnlyCatalog([string]$Cat) {
    # keep the catalog that was there (NVIDIA's own, or step 3's Inf2Cat one) as nv_disp.cat.full
    if ((Test-Path $Cat) -and -not (Test-Path "$Cat.full")) { Copy-Item $Cat "$Cat.full" }
    $cdf = Join-Path $env:TEMP 'p100-win-eac.cdf'
    $lines = @('[CatalogHeader]', "Name=$(Split-Path $Cat -Leaf)", "ResultDir=$(Split-Path $Cat)", 'PublicVersion=0x0000001',
               'CatalogVersion=2', 'HashAlgorithms=SHA256', 'EncodingType=0x00010001', 'CATATTR1=0x10010001:OSAttr:2:10.0',
               '', '[CatalogFiles]')
    $infs = Get-ChildItem $dir -Filter '*.inf'
    foreach ($inf in $infs) { $lines += "<hash>$($inf.Name)=$($inf.FullName)" }
    Set-Content -Path $cdf -Value $lines -Encoding ASCII
    Remove-Item $Cat -Force -ErrorAction SilentlyContinue
    & $makecat -v $cdf | Select-Object -Last 3
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Cat)) { throw "makecat failed ($LASTEXITCODE)" }
    Remove-Item $cdf
    Write-Host "catalog covers $($infs.Count) INFs, no binaries"
}

function Remove-NvidiaDisplayPackages {
    foreach ($p in Get-WindowsDriver -Online | Where-Object { $_.ProviderName -match 'NVIDIA' -and $_.ClassName -eq 'Display' }) {
        pnputil /delete-driver $p.Driver /uninstall /force
        if ($LASTEXITCODE -ne 0) { throw "pnputil /delete-driver $($p.Driver) failed ($LASTEXITCODE)" }
    }
}

# The "Windows can't verify the publisher of this driver software" dialog: a #32770 window with a command link
# labelled "Install this driver software anyway".
function Invoke-InstallAnyway {
    $desktop = [Windows.Automation.AutomationElement]::RootElement
    $dialogs = $desktop.FindAll([Windows.Automation.TreeScope]::Children,
        (New-Object Windows.Automation.PropertyCondition([Windows.Automation.AutomationElement]::ClassNameProperty, '#32770')))
    foreach ($d in $dialogs) {
        $buttons = $d.FindAll([Windows.Automation.TreeScope]::Descendants,
            (New-Object Windows.Automation.PropertyCondition([Windows.Automation.AutomationElement]::ControlTypeProperty,
                [Windows.Automation.ControlType]::Button)))
        foreach ($b in $buttons) {
            if ($b.Current.Name -like 'Install this driver software anyway*') {
                $b.GetCurrentPattern([Windows.Automation.InvokePattern]::Pattern).Invoke()
                Write-Host 'accepted the unsigned-driver prompt'
            }
        }
    }
}

function Install-Package {
    $proc = Start-Process pnputil.exe -ArgumentList '/add-driver', "`"$dir\nv_dispwi.inf`"", '/install' -PassThru `
        -RedirectStandardOutput $log -WindowStyle Hidden
    $null = $proc.Handle   # without an open handle, ExitCode reads empty after the process exits
    if ($AutoAccept) {
        Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
        # UI Automation queries time out while the display driver is being replaced; keep polling through them
        $uiaTimeouts = 0
        while (-not $proc.HasExited) {
            try { Invoke-InstallAnyway } catch [System.Runtime.InteropServices.COMException] { $uiaTimeouts++ }
            Start-Sleep -Milliseconds 500
        }
        if ($uiaTimeouts) { Write-Host "UI Automation timed out $uiaTimeouts times during the display restart" }
    }
    $proc.WaitForExit()
    Get-Content $log | Select-Object -Last 8
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        throw "pnputil /add-driver failed ($($proc.ExitCode)); rollback.ps1 reinstalls the backed-up driver"
    }
}

Write-Host '== 1. INF-only catalog'
$cat = Join-Path $dir 'nv_disp.cat'
New-InfOnlyCatalog $cat

Write-Host '== 2. new certificate, sign, delete its private key'
$previous = @(Get-Content $thumbFile -ErrorAction SilentlyContinue | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$cert = New-CatalogSigningCert $thumbFile $Subject
& $signtool sign /fd sha256 /sm /s My /sha1 $cert.Thumbprint $cat
if ($LASTEXITCODE -ne 0) { throw "signtool failed ($LASTEXITCODE)" }
Remove-Item "Cert:\LocalMachine\My\$($cert.Thumbprint)" -DeleteKey
Write-Host "signed by $($cert.Subject) ($($cert.Thumbprint), recorded in $thumbFile); private key deleted"

Write-Host '== 3. reinstall the package'
Remove-NvidiaDisplayPackages
Install-Package

Write-Host '== 4. remove the previous certificates'
Remove-CatalogSigningCert $previous $thumbFile
Write-Host "removed: $($previous -join ', ')"

Write-Host '== 5. P100 driver model (the reinstall reset it to TCC)'
& "$PSScriptRoot\3b_p100_driver_model.ps1"
Write-Host '== done. Reboot now, then run 4_verify.ps1.'
