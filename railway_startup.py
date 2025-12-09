# railway.startup.py - Startup script for Railway deployment

import os
import sys
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Change directory to the backend folder where the app is located
    backend_path = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_path)

    # Add the backend directory to the Python path
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    # Import and run the main app
    from app.main import app
    logger.info("Successfully imported main app")
except ImportError as e:
    logger.error(f"Failed to import main app: {str(e)}")
    # Create a fallback app for error handling
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"error": "Failed to load main application", "details": str(e)}

    @app.get("/health")
    async def health():
        return {"error": "Failed to load main application", "details": str(e)}
except Exception as e:
    logger.error(f"Unexpected error during startup: {str(e)}")
    # Create a fallback app for error handling
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"error": "Unexpected error during startup", "details": str(e)}

    @app.get("/health")
    async def health():
        return {"error": "Unexpected error during startup", "details": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            lifespan="on"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        import traceback
        traceback.print_exc()