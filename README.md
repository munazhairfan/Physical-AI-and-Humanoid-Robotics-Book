# Physical AI & Humanoid Robotics Educational Platform

An interactive educational platform on Physical AI & Humanoid Robotics with integrated RAG chatbot for enhanced learning.

## 📚 Project Structure

```
Physical-AI-and-Humanoid-Robotics/
├── backend/                    # RAG API service (FastAPI)
│   ├── app/                   # Main application code
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile            # Deployment configuration
│   ├── app.py               # Main entry point
│   └── startup.py           # Railway startup script
├── frontend/                 # Docusaurus documentation site
│   └── rag-chatbot-frontend/ # The actual Docusaurus site
│       ├── src/              # React components
│       │   └── components/   # Chat widget component
│       ├── docs/            # Book content
│       ├── package.json     # Frontend dependencies
│       ├── vercel.json      # Vercel deployment config
│       └── docusaurus.config.ts
├── .github/workflows/        # GitHub Actions workflows
│   └── gh-pages.yml         # GitHub Pages deployment
├── .env                     # Environment variables (not in repo)
└── DEPLOYMENT_GUIDE.md      # Detailed deployment instructions
```

## 🚀 Quick Start

### Backend (RAG API)
```bash
cd backend
pip install -r requirements.txt
# Set your GEMINI_API_KEY in environment
python app.py
```
Backend will run on `http://localhost:8000`

### Frontend (Documentation)
```bash
cd frontend/rag-chatbot-frontend
npm install
# Set REACT_APP_BACKEND_URL to your backend URL
npm start
```
Frontend will run on `http://localhost:3000`

## ☁️ GitHub Deployment

### Prerequisites

1. A GitHub account
2. Accounts on deployment platforms:
   - [Railway](https://railway.app) for backend deployment
   - [Vercel](https://vercel.com) for frontend deployment

### Backend Deployment (Railway)

1. **Prepare for Deployment**
   - Ensure all changes are committed to your GitHub repository
   - The backend is configured with a `Dockerfile` and `startup.py` for Railway deployment

2. **Deploy to Railway**
   - Go to [Railway](https://railway.app)
   - Connect your GitHub account
   - Create a new project and select this repository
   - Choose the `backend` directory as the root
   - Railway will automatically detect the Python project and use the Dockerfile
   - Add environment variable:
     - Key: `GEMINI_API_KEY`
     - Value: Your Google Gemini API key
   - Deploy the project

3. **Note Your Backend URL**
   - After deployment, Railway will provide a URL like: `https://your-app-name.up.railway.app`
   - Save this URL - you'll need it for frontend configuration

### Frontend Deployment (Vercel)

1. **Prepare for Deployment**
   - The frontend is already configured for Vercel deployment
   - baseUrl is set to `/` for Vercel root deployment

2. **Deploy to Vercel**
   - Go to [Vercel](https://vercel.com)
   - Connect your GitHub account
   - Create a new project and select this repository
   - Set the root directory to: `frontend/rag-chatbot-frontend`
   - Add environment variable:
     - Key: `REACT_APP_BACKEND_URL`
     - Value: Your Railway backend URL from the previous step
   - Configure build settings:
     - Build Command: `npm run build`
     - Output Directory: `build`
     - Install Command: `npm install`
   - Deploy the project

### Alternative: GitHub Pages Deployment

If you prefer to deploy the frontend on GitHub Pages:

1. The repository already includes a GitHub Actions workflow for GitHub Pages deployment
2. The workflow is configured in `.github/workflows/gh-pages.yml`
3. To enable GitHub Pages deployment:
   - Go to your repository settings
   - Navigate to "Pages" section
   - Select source as "GitHub Actions"
   - The workflow will automatically build and deploy the frontend on pushes to main branch

## 🤖 API Key Configuration

The application uses Google's Gemini API for AI responses. The GEMINI_API_KEY is required for the backend to function properly:

- **Backend**: Set as environment variable on Railway (`GEMINI_API_KEY`)
- **Security**: The API key is stored in environment variables and never committed to the repository
- **Local Development**: Create a `.env` file in the backend directory with your API key

## 🔧 Troubleshooting

### If Chat Widget Doesn't Connect
1. Check that your backend URL is correctly set in the frontend environment variables
2. Verify your backend is properly deployed and accessible
3. Check browser console for any connection errors

### If Backend Deployment Fails
1. Make sure Dockerfile doesn't have import tests during build
2. Verify Python version compatibility (should be 3.11+)
3. Check the backend logs in Railway dashboard

### If Frontend Doesn't Load Properly
1. Make sure baseUrl in docusaurus.config.ts is set to `/` for Vercel root deployment
2. After deployment, update the `url` field with your actual deployment domain
3. Verify that all static assets load correctly
4. Check browser console for any 404 errors or asset loading issues

### Common Issues Fixed
- ✅ **Railway Import Error**: Fixed Dockerfile to remove problematic import test
- ✅ **Vercel 404 Error**: Added proper routing configuration
- ✅ **BaseUrl Issue**: Fixed for Vercel root deployment
- ✅ **URL Configuration**: Updated for Vercel deployment
- ✅ **Duplicate Backends**: Consolidated to single, properly configured backend
- ✅ **Python Version Mismatch**: Updated to Python 3.11 consistently

## 🤝 Integration

The floating chat widget on the documentation site connects to the backend API to provide contextual answers about robotics concepts, creating an interactive learning experience.

## 📖 Features

- Interactive robotics textbook with integrated Q&A
- Select text to get explanations from AI
- Knowledge base for robotics concepts
- Conversational interface for learning
- Multi-modal support for complex robotics queries
- Real-time vector search for relevant context
- Responsive design for all devices