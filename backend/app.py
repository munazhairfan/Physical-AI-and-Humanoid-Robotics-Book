# app.py - EntryPoint for Railway deployment
# This file is used by Railway to find and start the application

import os
import sys
import uvicorn

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import the main app
from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        lifespan="on"
    )