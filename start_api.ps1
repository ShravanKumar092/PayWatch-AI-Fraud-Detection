# Start API Server Script
Write-Host "=== Starting PayWatch AI API Server ===" -ForegroundColor Green
Write-Host ""

# Check if port is already in use
$existing = Get-NetTCPConnection -LocalPort 8020 -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "WARNING: Port 8020 is already in use!" -ForegroundColor Red
    Write-Host "Please stop the existing server first or use restart_api.ps1" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Do you want to kill the existing process? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        $existing | Select-Object -ExpandProperty OwningProcess | ForEach-Object { 
            Stop-Process -Id $_ -Force
            Write-Host "Stopped process: $_" -ForegroundColor Gray
        }
        Start-Sleep -Seconds 2
    } else {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit
    }
}

# Ensure we're in the project root directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Start the server from project root (this ensures all imports work correctly)
Write-Host "Server starting on http://127.0.0.1:8020" -ForegroundColor Green
Write-Host "API Docs: http://127.0.0.1:8020/docs" -ForegroundColor Cyan
Write-Host "Health Check: http://127.0.0.1:8020/health" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
python -m uvicorn api.app:app --host 127.0.0.1 --port 8020 --reload

