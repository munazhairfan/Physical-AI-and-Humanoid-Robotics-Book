# api/index.py - Vercel API route for FastAPI app
import os
from app.main import app
from mangum import Mangum

# Set the environment variable for the port
os.environ.setdefault("PORT", "8000")

# Create Mangum handler for ASGI compatibility
handler = Mangum(app, lifespan="off")

# The handler function that Vercel will call
def index(event, context):
    """
    This is the main function that Vercel will call.
    The event and context parameters are provided by Vercel.
    """
    return handler(event, context)