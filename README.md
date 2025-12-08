# Physical AI & Humanoid Robotics Educational Book

An interactive educational book on Physical AI & Humanoid Robotics with integrated RAG chatbot for enhanced learning.

## 📚 Project Structure

```
Physical-AI-and-Humanoid-Robotics/
├── backend/                    # RAG API service (FastAPI)
│   ├── app/                   # Main application code
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile            # Deployment configuration
│   └── app.py               # Main entry point
├── frontend/                 # Docusaurus documentation site
│   └── rag-chatbot-frontend/ # The actual Docusaurus site
│       ├── src/              # React components
│       │   └── components/   # Chat widget component
│       ├── docs/            # Book content
│       ├── package.json     # Frontend dependencies
│       └── docusaurus.config.ts
```

## 🚀 Quick Start

### Backend (RAG API)
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Backend will run on `http://localhost:8000`

### Frontend (Documentation)
```bash
cd frontend/rag-chatbot-frontend
npm install
npm start
```
Frontend will run on `http://localhost:3000`

## ☁️ Deployment

### Backend to Railway
1. Push code to GitHub
2. Connect to Railway
3. Deploy `backend/` directory
4. Backend API URL will be provided by Railway

### Frontend to Vercel
1. Push code to GitHub
2. Connect to Vercel
3. Set build directory to `frontend/rag-chatbot-frontend`
4. Add environment variable:
   - `REACT_APP_BACKEND_URL`: Your deployed backend URL
5. **Important**: baseUrl is configured as `/` for Vercel root deployment
6. Deploy

## 🔧 Troubleshooting

### Common Issues Fixed
- ✅ **Railway Import Error**: Fixed Dockerfile to remove problematic import test
- ✅ **Vercel 404 Error**: Added proper routing configuration
- ✅ **BaseUrl Issue**: Fixed for Vercel root deployment
- ✅ **Duplicate Backends**: Consolidated to single, properly configured backend
- ✅ **Python Version Mismatch**: Updated to Python 3.11 consistently

## 🤝 Integration

The floating chat widget on the documentation site connects to the backend API to provide contextual answers about robotics concepts, creating an interactive learning experience.

## 📖 Features

- Interactive robotics textbook with integrated Q&A
- Select text to get explanations from AI
- Knowledge base for robotics concepts
- Conversational interface for learning