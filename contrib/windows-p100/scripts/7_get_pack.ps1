# Step 7: downloads the Qwen3.8-Flash-Next NVFP4 pack from Hugging Face into -Dest (resumable: rerun to continue) and
# checks its size and SHA-256. The file is 131.7 GB; put it on a fast local SSD. Of it, ~67.8 GiB of weights go to the
# GPUs and RAM at start; the n-gram table (~51 GiB) stays on the disk and is read while the model runs.
# (The test machine kept the pack on an NTFS partition that its Linux install also mounted, through ntfs3. That
# partition setup is specific to that machine and is not part of this script.)
param([Parameter(Mandatory = $true)][string]$Dest)
$ErrorActionPreference = 'Stop'
# Windows PowerShell's own module paths (PS7's, inherited when launched from pwsh, hide Get-FileHash)
$env:PSModulePath = "$env:ProgramFiles\WindowsPowerShell\Modules;$env:SystemRoot\system32\WindowsPowerShell\v1.0\Modules"
$PackUrl = 'https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt/resolve/main/Qwen3.8-FN-NVFP4-DQ6_K.basalt'
$PackName = 'Qwen3.8-FN-NVFP4-DQ6_K.basalt'
$PackBytes = 131681979197
$PackSha256 = '177f9b70c63ecd7efd17fc705ad85a8c8253d59e4f04dd10a001b556d15fad33'

New-Item -ItemType Directory -Force $Dest | Out-Null
$path = Join-Path $Dest $PackName
# curl reports progress on stderr, which Windows PowerShell would turn into a terminating error under 'Stop'
$ErrorActionPreference = 'Continue'
& curl.exe -L --fail --retry 5 --retry-all-errors -C - -o $path $PackUrl
$curlExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($curlExit -ne 0) { throw "download failed (curl exit $curlExit); rerun to resume" }
$size = (Get-Item $path).Length
if ($size -ne $PackBytes) { throw "pack is $size bytes, expected $PackBytes" }
$hash = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant()
if ($hash -ne $PackSha256) { throw "pack SHA-256 $hash does not match Hugging Face's $PackSha256" }
Write-Host "pack verified: $path, $PackBytes bytes, SHA-256 $hash"
