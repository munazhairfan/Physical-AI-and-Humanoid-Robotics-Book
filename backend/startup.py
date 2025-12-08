#!/usr/bin/env python
# startup.py - Hugging Face Spaces startup script

import os
import sys
from app.main import app

def main():
    """Main entry point for Hugging Face Spaces"""
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app",  # Load app from the main module
        host="0.0.0.0",
        port=port,
        workers=1
    )

if __name__ == "__main__":
    main()