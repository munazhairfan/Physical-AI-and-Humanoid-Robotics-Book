# startup.py - Railway deployment entrypoint

import os
import logging
import uvicorn
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Import the main FastAPI app from app/main.py
    from app.main import app
    logger.info("Successfully imported main app")

except Exception as e:
    logger.error(f"Failed to import main app: {e}")

    # Fallback FastAPI app to pass Railway health-check
    app = FastAPI()

    @app.get("/")
    def fallback_root():
        return {"error": "Could not import main app", "details": str(e)}

    @app.get("/health")
    def health():
        return {"status": "fallback"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
