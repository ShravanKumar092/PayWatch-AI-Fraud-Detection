@echo off
echo ========================================
echo Starting PayWatch AI Streamlit Dashboard
echo ========================================
echo.

REM Check if streamlit is available
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit not found. Installing...
    pip install streamlit
    echo.
)

echo Starting Streamlit on http://localhost:8501
echo Press Ctrl+C to stop the dashboard
echo.
python -m streamlit run app.py
pause

