# main.py - Main entry point for Railway deployment
import os
import sys
import importlib.util

# Add the backend directory to Python path so we can import the app
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)

# Import the main FastAPI app using importlib to ensure it works regardless of execution context
spec = importlib.util.spec_from_file_location("app", os.path.join(backend_dir, "app", "main.py"))
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
app = app_module.app

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        lifespan="on"
    )