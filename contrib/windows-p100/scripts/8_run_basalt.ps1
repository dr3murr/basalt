# Step 8: serves Qwen3.8-Flash-Next with Basalt on the 5080 + P100 tier, OpenAI/Anthropic-compatible API on
# http://127.0.0.1:8090. Writes the config on first use: the settings the test machine ran (int8 KV cache, resident
# experts, peer-exclusive on). With Windows' smaller free RAM, f14e879's automatic rule left the P100 mirroring RAM
# experts (measured: conv 67 t/s and a RAM-guard stop, against 108/89/101 t/s exclusive); this branch's auto plan
# (b1049ff) turns peer-exclusive on by itself whenever the experts do not all fit, and the config still sets it.
# Run it elevated for large pages. The server's Python packages go in $Src\.venv (README, step 6).
param(
    [Parameter(Mandatory = $true)][string]$Pack,   # the .basalt file from 7_get_pack.ps1
    [string]$Src = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path,
    [string]$Config = "$env:USERPROFILE\.config\basalt\flashnext.json",
    [string]$Cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
)
$ErrorActionPreference = 'Stop'
$env:PATH = "$Cuda\bin\x64;$env:PATH"   # basalt.exe loads cublas64_13.dll from there (cudart is linked statically)
if (-not (Test-Path $Pack)) { throw "pack not found: $Pack (run 7_get_pack.ps1)" }
if (-not (Test-Path $Config)) {
    New-Item -ItemType Directory -Force (Split-Path $Config) | Out-Null
    $json = [ordered]@{
        server = [ordered]@{ model = (Resolve-Path $Pack).Path; alias = 'qwen3.8-flashnext'; gpu = '0,1'; host = '127.0.0.1'; port = 8090;
            lazy = $true; 'idle-unload' = 1800; 'min-p' = 0.05; 'fit-max-tokens' = $true }
        engine = [ordered]@{ 'max-context' = 262144; kv = 'int8'; 'resident-experts' = $true; 'vram-reserve-mib' = 700;
            'peer-exclusive' = $true }
    } | ConvertTo-Json -Depth 3
    # UTF-8 without a byte-order mark: Set-Content -Encoding UTF8 writes one, and the server's JSON parser rejects it
    [IO.File]::WriteAllText($Config, $json, (New-Object Text.UTF8Encoding $false))
    Write-Host "wrote $Config"
}
Set-Location $Src
& "$Src\bin\basalt.exe" serve --config $Config
