# app.py - Entry point for Railway deployment
# This creates the FastAPI app instance that Railway expects

import os
import sys
import logging
import uvicorn

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory of this file and add the app subdirectory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Import and handle any potential startup issues gracefully
startup_error = None
try:
    # Import the main FastAPI application - use absolute import to avoid relative import issues
    from app.main import app  # Import from the main module inside the app directory
    logger.info("Successfully imported main app")
except Exception as e:
    startup_error = str(e)
    logger.error(f"Error importing main app: {startup_error}")
    # Create a fallback app for error handling
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"error": "Failed to load main application", "details": startup_error}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        lifespan="on"
    )