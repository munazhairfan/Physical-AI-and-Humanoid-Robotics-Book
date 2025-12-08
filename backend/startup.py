#!/usr/bin/env python
# startup.py - Railway/production startup script

import os
import sys

def main():
    """Main entry point for Railway deployment"""
    # Add current directory to Python path to ensure imports work
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    # Import uvicorn after dependencies are available
    import uvicorn

    # Import the main app - use the approach that matches app.py
    try:
        from app import app  # Import from the main app.py entry point
    except ImportError:
        # Fallback: try direct import from app.main
        try:
            from app.main import app
        except ImportError as e:
            print(f"Could not import app: {e}")
            raise

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