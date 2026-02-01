@echo off
echo ========================================
echo Starting PayWatch AI API Server
echo ========================================
echo.

REM Check if port 8010 is already in use
netstat -ano | findstr :8010 | findstr LISTENING >nul
if %errorlevel% == 0 (
    echo WARNING: Port 8010 is already in use!
    echo.
    echo To stop the existing server, run:
    echo   stop_api.bat
    echo.
    pause
    exit /b 1
)

echo Checking dependencies...
python -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo ERROR: Missing dependencies. Installing...
    pip install -r Requirements.txt
    echo.
)

echo.
cd api
echo Starting server on http://127.0.0.1:8010
echo Press Ctrl+C to stop the server
echo.
python -m uvicorn app:app --host 127.0.0.1 --port 8010 --reload
pause

