#Requires -RunAsAdministrator
# Runs step 3 (which ends with 3b, the P100 driver model) as SYSTEM via start_unattended.ps1, so closing the terminal
# (or losing the display while the driver changes) cannot kill it. Always reboots: the P100's driver model takes
# effect at boot. Before the reboot it registers a one-shot startup task that writes 4_verify.ps1's output to
# $StateDir\verify.log. Everything else is logged to $StateDir\unattended.log (by the task action).
param(
    [Parameter(Mandatory = $true)][string]$DriverDir,
    [string]$StateDir = (Join-Path $env:ProgramData 'p100-win')
)
$ErrorActionPreference = 'Stop'
$env:PSModulePath = "$env:ProgramFiles\WindowsPowerShell\Modules;$env:SystemRoot\system32\WindowsPowerShell\v1.0\Modules"
Write-Host "=== $(Get-Date -Format s) unattended install start"

& "$PSScriptRoot\3_install_r580.ps1" -DriverDir $DriverDir -StateDir $StateDir

Get-PnpDevice | Where-Object { $_.InstanceId -match 'VEN_10DE&DEV_(2C02|15F8)' } |
    Select-Object FriendlyName, Status, Problem | Format-Table -AutoSize | Out-String | Write-Host
$postBoot = "& '$PSScriptRoot\4_verify.ps1'"

$cmd = "$postBoot *> '$StateDir\verify.log'; Unregister-ScheduledTask -TaskName p100-win-verify -Confirm:`$false"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$cmd`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = 'PT90S'
Register-ScheduledTask -TaskName 'p100-win-verify' -Action $action -Trigger $trigger -User 'SYSTEM' -RunLevel Highest -Force | Out-Null
Write-Host "=== $(Get-Date -Format s) rebooting in 60 s; verify runs 90 s after boot into $StateDir\verify.log"
shutdown.exe /r /t 60 /c 'p100-win: R580 driver installed, rebooting (unattended)'
