# Deployment Guide

## Backend Deployment (Railway)

### Step 1: Prepare Backend
- Ensure your backend is in the `backend/` directory
- The Dockerfile is already configured for Railway deployment
- Make sure all changes are committed to your repository

### Step 2: Deploy to Railway
1. Go to [Railway](https://railway.app)
2. Connect your GitHub repository
3. Create a new project
4. Select the `backend` directory as the root
5. Railway will automatically detect the Python project and use the Dockerfile
6. Deploy

### Step 3: Note Your Backend URL
After deployment, Railway will provide a URL like: `https://your-app-name.up.railway.app`
Save this URL - you'll need it for frontend configuration.

## Frontend Deployment (Vercel)

### Step 1: Prepare Frontend
- Ensure frontend is in `frontend/rag-chatbot-frontend/` directory
- The project is already configured for Vercel deployment
- **Important**: baseUrl is set to `/Physical-AI-and-Humanoid-Robotics-Book/` (following Vercel's suggestion)

### Step 2: Deploy to Vercel
1. Go to [Vercel](https://vercel.com)
2. Connect your GitHub repository
3. Create a new project
4. Set the root directory to: `frontend/rag-chatbot-frontend`
5. Add environment variable:
   - Key: `REACT_APP_BACKEND_URL`
   - Value: Your Railway backend URL from Step 3 above
6. Configure build settings:
   - Build Command: `npm run build`
   - Output Directory: `build`
   - Install Command: `npm install`

### Step 3: Complete Integration
- Frontend will be deployed to a Vercel URL
- The floating chat widget will connect to your backend automatically
- Both services will work together to provide the interactive learning experience

## Verification

After both deployments:
1. Visit your frontend URL
2. The floating chat widget should connect to your backend
3. You should be able to ask questions about robotics concepts
4. Selecting text on pages should allow you to ask about that content

## Troubleshooting

### If Chat Widget Doesn't Connect
1. Check that your backend URL is correctly set in Vercel environment variables
2. Verify your backend is properly deployed and accessible
3. Check browser console for any connection errors

### If Backend Deployment Fails
1. Make sure Dockerfile doesn't have import tests during build
2. Verify Python version compatibility (should be 3.11+)
3. Check the backend logs in Railway dashboard

### If Frontend Doesn't Load Properly
1. Check if Vercel suggests a specific baseUrl (e.g., "/project-name/")
2. Update baseUrl in docusaurus.config.ts accordingly
3. Verify that all static assets load correctly
4. Check browser console for any 404 errors or asset loading issues