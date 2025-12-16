# Physical AI & Humanoid Robotics - Project Architecture

## Overview
Educational book on Physical AI & Humanoid Robotics with integrated RAG chatbot for interactive learning.

## Architecture

### Backend Service (`/backend`)
- **Technology**: FastAPI Python server
- **Purpose**: RAG (Retrieval Augmented Generation) API
- **Endpoints**: 
  - `/chat` - Interactive Q&A with robotics knowledge
  - `/embed` - Text embedding capabilities
  - `/query` - Document search functionality
  - `/ingest` - Document ingestion into knowledge base
  - `/selected_text` - Process selected text from documentation
- **Deployment**: Railway (or similar cloud platform)

### Frontend Documentation Site (`/frontend/rag-chatbot-frontend`)
- **Technology**: Docusaurus React static site
- **Purpose**: Educational book interface with integrated chat
- **Features**:
  - Course content documentation
  - Interactive floating chat widget
  - Text selection to ask questions about content
- **Deployment**: Vercel (or similar static site platform)

## Integration

The floating chat widget in the documentation connects to the deployed backend API to provide contextual answers about robotics concepts.

## Recommended Deployment

1. Deploy backend to Railway:
   - Backend URL: `https://your-app.up.railway.app`
   - Set in frontend environment variables

2. Deploy frontend to Vercel:
   - Set `REACT_APP_BACKEND_URL` to your deployed backend URL
   - Frontend will connect to backend API for chat functionality

## Development

- Backend: `cd backend && python app.py`
- Frontend: `cd frontend/rag-chatbot-frontend && npm start`
- Both must run simultaneously for full functionality