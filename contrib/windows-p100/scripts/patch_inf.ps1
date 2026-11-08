# Step 2b: adds the kit's two devices to YOUR copy of the NVIDIA Data Center 582.99 package (extract it with 7-Zip;
# -DriverDir is its Display.Driver folder). nv_dispwi.inf gets, at the end of [NVIDIA_Devices.NTamd64.10.0...17134]
# (after the RTX PRO 4000 Blackwell line) and in [Strings]:
#   %NVIDIA_DEV.2C02% = Section094, PCI\VEN_10DE&DEV_2C02    RTX 5080, in the section 582.99 uses for the
#                                                             RTX PRO 4500/4000 Blackwell (the same GB203 die)
#   %NVIDIA_DEV.15F8% = Section026, PCI\VEN_10DE&DEV_15F8    any Tesla P100-PCIE-16GB; the stock line matches only
#                                                             subsystem 118F10DE, Section026 is the stock P100 section
#   NVIDIA_DEV.2C02 = "NVIDIA GeForce RTX 5080"
#   NVIDIA_DEV.15F8 = "NVIDIA Tesla P100-PCIE-16GB"
# Byte-exact (CRLF and the column padding kept): the stock INF, SHA-256 080b2285..., becomes the file the test
# machine installed, SHA-256 c0a840e3... Any other input is refused unless -Force. The original is saved as
# nv_dispwi.inf.orig next to the Display.Driver folder, not inside it (the catalog step hashes that folder).
# Changes no system state; no administrator rights needed.
param(
    [Parameter(Mandatory = $true)][string]$DriverDir,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$StockSha256 = '080b2285660112ffdeb2d4fbd35886ca9a69febb725074f5349d8e18e0c66e4d'
$PatchedSha256 = 'c0a840e3674b4c64f0a9de85afd7c958351bea89c609fef9b7f4cc47d308d6fa'
$ModelAnchor = '%NVIDIA_DEV.2C34.2052.10DE%'
$StringAnchor = 'NVIDIA_DEV.2C34.2052.10DE ='
$ModelLines = @(
    '%NVIDIA_DEV.2C02%           = Section094, PCI\VEN_10DE&DEV_2C02                 ',
    '%NVIDIA_DEV.15F8%           = Section026, PCI\VEN_10DE&DEV_15F8                 ')
$StringLines = @(
    'NVIDIA_DEV.2C02 = "NVIDIA GeForce RTX 5080"',
    'NVIDIA_DEV.15F8 = "NVIDIA Tesla P100-PCIE-16GB"')
$Latin1 = [Text.Encoding]::GetEncoding(28591)   # one character per byte: the file is written back byte for byte

function Get-Sha256([byte[]]$Bytes) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { ([BitConverter]::ToString($sha.ComputeHash($Bytes)) -replace '-', '').ToLowerInvariant() } finally { $sha.Dispose() }
}

$inf = Join-Path $DriverDir 'nv_dispwi.inf'
if (-not (Test-Path $inf)) { throw "$inf not found: -DriverDir is the extracted package's Display.Driver folder" }
$bytes = [IO.File]::ReadAllBytes((Resolve-Path $inf).Path)
$hash = Get-Sha256 $bytes
if ($hash -eq $PatchedSha256) { Write-Host "already patched: $inf"; return }
if ($hash -ne $StockSha256 -and -not $Force) {
    throw "$inf has SHA-256 $hash, not the stock 582.99 INF ($StockSha256): another package, or edited already. -Force patches it anyway."
}
$lines = New-Object 'System.Collections.Generic.List[string]'
$lines.AddRange([string[]]($Latin1.GetString($bytes) -split "`r`n"))
foreach ($l in $lines) {
    if ($l -match '^%NVIDIA_DEV\.(2C02|15F8)%') { throw "$inf already has a DEV_$($Matches[1]) line without a subsystem: nothing done" }
}
function Add-LinesAfter([string]$Prefix, [string[]]$New) {
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i].StartsWith($Prefix, [StringComparison]::Ordinal)) { $lines.InsertRange($i + 1, $New); return }
    }
    throw "line starting with '$Prefix' not found in ${inf}: nothing done"
}
Add-LinesAfter $ModelAnchor $ModelLines
Add-LinesAfter $StringAnchor $StringLines
$out = $Latin1.GetBytes(($lines -join "`r`n"))
$new = Get-Sha256 $out
if ($hash -eq $StockSha256 -and $new -ne $PatchedSha256) { throw "the result has SHA-256 $new, expected ${PatchedSha256}: nothing written" }
$folder = (Resolve-Path $DriverDir).Path
$backup = Join-Path (Split-Path $folder -Parent) 'nv_dispwi.inf.orig'
if (-not (Test-Path $backup)) { [IO.File]::WriteAllBytes($backup, $bytes) }
[IO.File]::WriteAllBytes((Join-Path $folder 'nv_dispwi.inf'), $out)
Write-Host "patched $inf (SHA-256 $new); original kept as $backup"
