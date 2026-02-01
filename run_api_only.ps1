# Start PayWatch AI - API Server Only
# (Streamlit has import issues, but API works perfectly)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting PayWatch AI - API Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$venv = "C:/Users/nnshr/OneDrive/Desktop/Fraud_Detection/Fraud_Detection/.venv/Scripts/python.exe"
$projectDir = "C:\Users\nnshr\OneDrive\Desktop\Fraud_Detection\Fraud_Detection"

# Kill existing processes on port 8020
Write-Host "Cleaning up existing processes..." -ForegroundColor Yellow
Get-NetTCPConnection -LocalPort 8020 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object { 
    Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1

# Start API Server
Write-Host "Starting API Server on port 8020..." -ForegroundColor Green
& $venv -m uvicorn api.app:app --host 127.0.0.1 --port 8020 --reload

Write-Host ""
Write-Host "API Server stopped." -ForegroundColor Yellow
