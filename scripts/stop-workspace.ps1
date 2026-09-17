param([int]$Port = 8010)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$expectedPython = [System.IO.Path]::GetFullPath((Join-Path $projectDirectory '.venv\Scripts\python.exe'))
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -eq 0) { Write-Output "No server is listening on port $Port."; return }
foreach ($listener in $listeners | Select-Object -ExpandProperty OwningProcess -Unique) {
    $serverProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $listener"
    $parentProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($serverProcess.ParentProcessId)"
    # Windows venv python may launch the base interpreter as the actual listener.
    $fromProject = $serverProcess.ExecutablePath -eq $expectedPython -or (
        $parentProcess.ExecutablePath -eq $expectedPython -and
        $parentProcess.CommandLine -match 'enterprise_agent\.cli\.serve' -and
        $parentProcess.CommandLine -match "--port\s+$Port(?:\s|$)"
    )
    if (-not $fromProject -or
        $serverProcess.CommandLine -notmatch 'enterprise_agent\.cli\.serve' -or
        $serverProcess.CommandLine -notmatch "--port\s+$Port(?:\s|$)" -or
        $serverProcess.CommandLine -notmatch '--mode\s+demo(?:\s|$)') {
        throw "Port $Port belongs to another process; refusing to stop it."
    }
    Stop-Process -Id $listener
    Write-Output "Stopped local demo workspace PID $listener. Persisted documents were not deleted."
}
