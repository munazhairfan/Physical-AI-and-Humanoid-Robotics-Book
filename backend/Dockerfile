FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x startup.py

# Test that the app can be imported - adjust the import path
RUN python -c "import sys; sys.path.insert(0, '.'); from app.main import app; print('Import successful')"

EXPOSE 8000

# Use the startup script
CMD ["python", "startup.py"]