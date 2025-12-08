#!/usr/bin/env python
# startup.py - Railway/production startup script

import os
import sys

def main():
    """Main entry point for Railway deployment"""
    # Add current directory to Python path
    sys.path.insert(0, '/app')

    # Import uvicorn after dependencies are available
    import uvicorn

    # Import the app module using importlib to handle potential import issues
    try:
        # Direct import approach
        from app.main import app
    except ImportError as e:
        # Fallback: Add the app directory to path and retry
        sys.path.insert(0, '/app/app')
        from main import app
        print(f"Direct import failed with: {e}. Used fallback import.")

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