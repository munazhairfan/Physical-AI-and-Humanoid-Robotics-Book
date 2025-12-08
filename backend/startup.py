#!/usr/bin/env python
# startup.py - Railway/production startup script

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, '.')

# Import the main FastAPI app
from app.main import app

def main():
    """Main entry point for Railway deployment"""
    import uvicorn

    # Get the port from environment variable (required by Railway)
    port = int(os.environ.get("PORT", 8000))

    print(f"Starting server on port {port}")

    # Run the application with uvicorn
    uvicorn.run(
        app,  # Pass the app instance directly
        host="0.0.0.0",
        port=port,
        workers=1,
        lifespan="on"
    )

if __name__ == "__main__":
    main()