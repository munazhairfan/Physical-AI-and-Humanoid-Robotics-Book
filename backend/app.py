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
try:
    # Import the main FastAPI application
    from main import app  # Import from the main module inside the app directory
    logger.info("Successfully imported main app")
except Exception as e:
    logger.error(f"Error importing main app: {str(e)}")
    # Create a fallback app for error handling
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"error": "Failed to load main application", "details": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        lifespan="on"
    )