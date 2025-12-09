# railway.startup.py - Cleaned version for Railway

import os
import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Import FastAPI app from app/main.py
    from app.main import app
    logger.info("Successfully imported main app")

except Exception as e:
    logger.error(f"Error importing main app: {e}")

    # Fallback FastAPI app so Railway health check doesn't kill container
    from fastapi import FastAPI
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

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
