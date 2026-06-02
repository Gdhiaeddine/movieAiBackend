$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $here

if (-not (Test-Path -LiteralPath ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}

$py = Join-Path $here ".venv\Scripts\python.exe"

Write-Host "Installing dependencies..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt

Write-Host "Starting FastAPI on http://0.0.0.0:8000 ..." -ForegroundColor Green
& $py -m uvicorn app.main:app --host 0.0.0.0 --port 8000
