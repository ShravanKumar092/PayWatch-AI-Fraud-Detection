# PowerShell script to start Streamlit dashboard
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting PayWatch AI Streamlit Dashboard" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if streamlit is available
try {
    python -c "import streamlit" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Streamlit not found. Installing..." -ForegroundColor Yellow
        pip install streamlit
        Write-Host ""
    }
} catch {
    Write-Host "Installing streamlit..." -ForegroundColor Yellow
    pip install streamlit
    Write-Host ""
}

Write-Host "Starting Streamlit on http://localhost:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Yellow
Write-Host ""

python -m streamlit run app.py

