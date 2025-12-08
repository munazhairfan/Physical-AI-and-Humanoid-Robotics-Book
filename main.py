# main.py - Main entry point for Railway deployment
import os
import sys
import uvicorn

# Add the backend directory to Python path so we can import the app
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)

# Change to the backend directory
os.chdir(backend_dir)

# Import the main FastAPI app
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