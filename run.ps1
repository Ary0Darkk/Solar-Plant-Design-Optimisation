# PowerShell runner for Solar Plant Optimization
$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $pythonPath) {
    & $pythonPath (Join-Path $PSScriptRoot "main.py") @args
} else {
    uv run python (Join-Path $PSScriptRoot "main.py") @args
}
