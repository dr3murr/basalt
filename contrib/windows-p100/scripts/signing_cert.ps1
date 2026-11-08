# Shared helpers, dot-sourced by 3_install_r580.ps1, 3e_reinstall_eac.ps1 and rollback.ps1.
#
# New-CatalogSigningCert creates a self-signed catalog-signing certificate (RSA 2048, SHA-256, non-exportable key,
# valid 5 years, the EKUs NVCleanstall uses: Windows Hardware Driver Extended Verification, Windows Hardware Driver
# Verification, Code Signing) in LocalMachine\My and trusts its public part in LocalMachine\Root and
# LocalMachine\TrustedPublisher. The caller signs the catalog and then deletes the private key. Every thumbprint is
# appended to the thumbprint file so rollback.ps1 can remove the certificate again. Read the README section
# "What trusting a self-made root certificate means" before using it.

# Default subject. The test machine used NVCleanstall's subject string instead (pass it with -Subject):
#   'CN=NVCleanstall Driver Modding Authority, O="https://www.techpowerup.com"'
# so that tools which already know NVCleanstall-modded drivers see a familiar issuer. That is only a name: such a
# certificate is made on your machine and has nothing to do with TechPowerUp.
$DefaultCatalogCertSubject = 'CN=p100-win local driver catalog signer (self-signed)'
$CatalogCertEku = '2.5.29.37={text}1.3.6.1.4.1.311.10.3.39,1.3.6.1.4.1.311.10.3.5,1.3.6.1.5.5.7.3.3'
$CatalogCertValidityYears = 5

function New-CatalogSigningCert([string]$ThumbprintFile, [string]$Subject) {
    if (-not $ThumbprintFile) { throw 'New-CatalogSigningCert: no thumbprint file given' }
    if (-not $Subject) { $Subject = $DefaultCatalogCertSubject }
    $cert = New-SelfSignedCertificate -Type Custom -Subject $Subject -TextExtension @($CatalogCertEku) `
        -KeyAlgorithm RSA -KeyLength 2048 -HashAlgorithm SHA256 -KeyExportPolicy NonExportable `
        -NotBefore (Get-Date).AddDays(-1) -NotAfter (Get-Date).AddYears($CatalogCertValidityYears) `
        -CertStoreLocation Cert:\LocalMachine\My
    New-Item -ItemType Directory -Force (Split-Path $ThumbprintFile) | Out-Null
    Add-Content -Path $ThumbprintFile -Value $cert.Thumbprint
    $cer = Join-Path $env:TEMP "catalog-signing-$($cert.Thumbprint).cer"
    Export-Certificate -Cert $cert -FilePath $cer | Out-Null
    Import-Certificate -FilePath $cer -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
    Import-Certificate -FilePath $cer -CertStoreLocation Cert:\LocalMachine\TrustedPublisher | Out-Null
    Remove-Item $cer
    $cert
}

# Removes certificates this kit created from all three stores (and the private key, if still present), then from the
# thumbprint file. Throws if a certificate cannot be removed, leaving the file intact.
function Remove-CatalogSigningCert([string[]]$Thumbprints, [string]$ThumbprintFile) {
    foreach ($t in $Thumbprints) {
        foreach ($store in 'Root', 'TrustedPublisher', 'My') {
            $path = "Cert:\LocalMachine\$store\$t"
            # -DeleteKey is a certificate-provider parameter: it binds only with -Path, not with pipeline input
            if (Test-Path $path) { Remove-Item -Path $path -DeleteKey -ErrorAction Stop }
        }
    }
    $keep = @(Get-Content $ThumbprintFile -ErrorAction SilentlyContinue | ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and $Thumbprints -notcontains $_ })
    Set-Content -Path $ThumbprintFile -Value $keep
}

# The newest copy of a Windows SDK / WDK tool under Windows Kits\10\bin (x64 first, then x86), or the given path.
function Find-KitTool([string]$Name, [string]$Given) {
    if ($Given) {
        if (-not (Test-Path $Given)) { throw "$Name not found at $Given" }
        return $Given
    }
    foreach ($arch in 'x64', 'x86') {
        $hit = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\$arch\$Name" -ErrorAction SilentlyContinue |
            Sort-Object FullName | Select-Object -Last 1
        if ($hit) { return $hit.FullName }
    }
    throw "$Name not found under ${env:ProgramFiles(x86)}\Windows Kits\10\bin: install the Windows SDK (signtool, makecat) or the WDK (Inf2Cat), or pass its path"
}

# Whether the package's nv_dispwi.inf carries the kit's RTX 5080 line (patch_inf.ps1).
function Test-PatchedInf([string]$DriverDir) {
    $inf = Join-Path $DriverDir 'nv_dispwi.inf'
    (Test-Path $inf) -and (Select-String -Path $inf -Pattern '^%NVIDIA_DEV\.2C02%\s*=\s*Section094,' -Quiet)
}
