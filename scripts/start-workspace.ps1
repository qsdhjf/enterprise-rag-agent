param(
    [int]$Port = 8010,
    [switch]$EnableModel,
    [switch]$EnableSetup
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw 'Run uv sync --locked in the project directory first.'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. No existing process was stopped."
}
$runtimeDirectory = Join-Path $projectDirectory '.local\runtime'
New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$stdoutPath = Join-Path $runtimeDirectory "workspace-$stamp.stdout.log"
$stderrPath = Join-Path $runtimeDirectory "workspace-$stamp.stderr.log"
$arguments = @('-B', '-m', 'enterprise_agent.cli.serve', '--mode', 'demo', '--host', '127.0.0.1', '--port', "$Port")
if ($EnableModel) { $arguments += '--enable-knowledge-model' }
if ($EnableSetup) { $arguments += '--enable-setup' }
$server = Start-Process -FilePath $pythonExecutable -ArgumentList $arguments -WorkingDirectory $projectDirectory -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if ($server.HasExited) { throw "Server exited. Read $stderrPath" }
    try {
        $config = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/config" -TimeoutSec 1
        if ($config.version) {
            Write-Output "Ready: http://127.0.0.1:$Port/ (PID $($server.Id), version $($config.version))"
            Write-Output "Logs: $stderrPath"
            Write-Output 'Runs in the background until stopped or Windows shuts down; this does not install an auto-start service.'
            return
        }
    } catch { }
    Start-Sleep -Milliseconds 250
}
throw "Server started but not ready yet. Inspect PID $($server.Id) and $stderrPath; it was not killed."
