#!/bin/bash

echo "========================================"
echo "Starting PayWatch AI API Server"
echo "========================================"
echo ""

# Check dependencies
python -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Missing dependencies. Installing..."
    pip install -r Requirements.txt
    echo ""
fi

cd api
echo "Starting server on http://127.0.0.1:8010"
echo "Press Ctrl+C to stop the server"
echo ""
python -m uvicorn app:app --host 127.0.0.1 --port 8010 --reload

