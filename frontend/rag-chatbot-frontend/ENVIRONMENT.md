# Environment Configuration

## Backend URL Setup

To connect the frontend chat widget to your backend API, set the appropriate environment variable:

### For Local Development
```bash
# In frontend/rag-chatbot-frontend/ directory
echo "REACT_APP_BACKEND_URL=http://localhost:8000" > .env
```

### For Production Deployment
Set the environment variable in your hosting platform:
- **Vercel**: Add `REACT_APP_BACKEND_URL` in project settings
- **Value**: Your deployed backend URL (e.g., `https://your-railway-app.up.railway.app`)

## API Endpoints Used
The chat widget connects to these backend endpoints:
- `/chat` - General Q&A
- `/selected_text` - Selected text processing
- `/book_query` - Book-specific queries

Make sure your backend is deployed and accessible at the configured URL.