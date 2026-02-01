# Start PayWatch AI - Fraud Detection System
# Runs both API Server and Streamlit Dashboard

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting PayWatch AI Fraud Detection" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$venv = "C:/Users/nnshr/OneDrive/Desktop/Fraud_Detection/Fraud_Detection/.venv/Scripts/python.exe"
$projectDir = "C:\Users\nnshr\OneDrive\Desktop\Fraud_Detection\Fraud_Detection"

# Kill existing processes on ports 8020 and 8501
Write-Host "Cleaning up existing processes..." -ForegroundColor Yellow
Get-NetTCPConnection -LocalPort 8020 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object { 
    Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
}
Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object { 
    Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1

# Start API Server in background
Write-Host "Starting API Server on port 8020..." -ForegroundColor Green
$apiProcess = Start-Process -FilePath $venv -ArgumentList "-m uvicorn api.app:app --host 127.0.0.1 --port 8020" -WorkingDirectory $projectDir -NoNewWindow -PassThru
Start-Sleep -Seconds 3

# Start Streamlit Dashboard in background
Write-Host "Starting Streamlit Dashboard on port 8501..." -ForegroundColor Green
$streamlitProcess = Start-Process -FilePath $venv -ArgumentList "-m streamlit run app.py --server.port 8501" -WorkingDirectory $projectDir -NoNewWindow -PassThru
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PayWatch AI is now running!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "API Server:" -ForegroundColor Yellow
Write-Host "  URL: http://127.0.0.1:8020" -ForegroundColor Cyan
Write-Host "  Docs: http://127.0.0.1:8020/docs" -ForegroundColor Cyan
Write-Host "  Health: http://127.0.0.1:8020/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Streamlit Dashboard:" -ForegroundColor Yellow
Write-Host "  URL: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# Wait for processes
$apiProcess, $streamlitProcess | Wait-Process

Write-Host ""
Write-Host "Services stopped." -ForegroundColor Yellow
