# Hugging Face Space app entry point
from app.main import app

# Ensure the FastAPI app is available at the module level
# This allows Hugging Face to import it as 'app'
asgi_app = app

# For backward compatibility and Hugging Face Spaces
__all__ = ['app', 'asgi_app']