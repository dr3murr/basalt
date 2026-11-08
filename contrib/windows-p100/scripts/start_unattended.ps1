#Requires -RunAsAdministrator
# Starts run_unattended.ps1 as a one-time SYSTEM scheduled task (independent of this terminal). It reboots the machine.
# Usage: start_unattended.ps1 -DriverDir <Display.Driver folder>
# Follow it with: Get-Content "$env:ProgramData\p100-win\unattended.log" -Wait
param(
    [Parameter(Mandatory = $true)][string]$DriverDir,
    [string]$StateDir = (Join-Path $env:ProgramData 'p100-win')
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path (Join-Path $StateDir 'packages.csv'))) { throw 'no verified backup: run 1_backup.ps1 first' }
$dir = (Resolve-Path $DriverDir).Path
$cmd = "& '$PSScriptRoot\run_unattended.ps1' -DriverDir '$dir' -StateDir '$StateDir' *>> '$StateDir\unattended.log'; Unregister-ScheduledTask -TaskName p100-win-install -Confirm:`$false"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$cmd`""
Register-ScheduledTask -TaskName 'p100-win-install' -Action $action -User 'SYSTEM' -RunLevel Highest -Force | Out-Null
Start-ScheduledTask -TaskName 'p100-win-install'
Write-Host "started task p100-win-install; log: $StateDir\unattended.log"
