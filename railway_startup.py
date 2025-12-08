# railway.startup.py - Startup script for Railway deployment

import os
import sys
import uvicorn

# Change directory to the backend folder where the app is located
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
os.chdir(backend_path)

# Add the backend directory to the Python path
sys.path.insert(0, backend_path)

# Import and run the main app
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