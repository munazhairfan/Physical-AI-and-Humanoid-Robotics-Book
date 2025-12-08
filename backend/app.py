# app.py - EntryPoint for Railway deployment
# This file is used by Railway to find and start the application

import os
import sys
import uvicorn

# Get the directory of this file and add the app subdirectory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Import the main app
from main import app  # Import from the main module inside the app directory

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        lifespan="on"
    )