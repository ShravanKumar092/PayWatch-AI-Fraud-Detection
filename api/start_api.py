#!/usr/bin/env python
import uvicorn
if __name__ == "__main__":
    print("Starting Fraud Detection API on port 8020...")
    uvicorn.run("app:app", host="127.0.0.1", port=8020, log_level="info")
