# Asterline Research Assistant - Single Launch Script for Windows PowerShell
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting Asterline Research Assistant..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$VenvPython = ".\.venv\Scripts\python.exe"
$VenvStreamlit = ".\.venv\Scripts\streamlit.exe"

if (-not (Test-Path ".venv")) {
    Write-Host "Virtual environment not detected. Creating .venv..." -ForegroundColor Yellow
    python -m venv --system-site-packages .venv
}

if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from template .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
}

Write-Host "Validating corpus integrity..." -ForegroundColor Green
& $VenvPython scripts/validate_corpus.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Corpus validation failed! Please check corpus.jsonl." -ForegroundColor Red
    exit 1
}

Write-Host "`nLaunching Streamlit Application..." -ForegroundColor Green
Write-Host "Opening http://localhost:8501 in your default browser..." -ForegroundColor Green
& $VenvStreamlit run app.py
