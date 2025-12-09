#!/usr/bin/env python
# startup.py - Railway/production startup script

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for Railway deployment"""
    # Add current directory to Python path to ensure imports work
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        # Import uvicorn after dependencies are available
        import uvicorn

        # Import the main app - use the approach that matches app.py
        try:
            from app import app  # Import from the main app.py entry point
        except ImportError as e:
            logger.error(f"Could not import from app: {e}")
            # Fallback: try direct import from app.main
            try:
                from app.main import app
            except ImportError as e2:
                logger.error(f"Could not import from app.main: {e2}")
                # Create a fallback app for error handling
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/")
                async def root():
                    return {"error": "Failed to load main application", "details": str(e2)}

                @app.get("/health")
                async def health():
                    return {"status": "degraded", "error": "Failed to load main application", "details": str(e2)}

        # Get the port from environment variable (required by Railway)
        port = int(os.environ.get("PORT", 8000))

        logger.info(f"Starting server on port {port}")

        # Run the application with uvicorn
        uvicorn.run(
            app,  # Pass the app instance directly
            host="0.0.0.0",
            port=port,
            workers=1,
            lifespan="on"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()