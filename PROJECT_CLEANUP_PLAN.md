# Project Structure Clean-up Summary

## Current Issues (with recommended fixes)

### 1. Duplicate Backend Implementation
- **Issue**: Two similar backend directories exist - `backend/` and `rag-chatbot-backend/`
- **Recommended Fix**: Keep only `backend/` as it's the more mature implementation with Railway fixes
- **Action**: Remove `rag-chatbot-backend/` directory (manually outside of this tool if needed)

### 2. Conflicting Root Directories  
- **Issue**: `src/` and `static/` at root level conflict with Docusaurus project structure
- **Recommended Fix**: Remove these directories as they're not properly organized
- **Action**: Move any necessary content to proper locations in `frontend/rag-chatbot-frontend/` if needed

### 3. Disorganized Content
- **Issue**: `sample_robotics_content.txt` in root directory
- **Recommended Fix**: Move to appropriate documentation location
- **Action**: Move to docs/ or remove if no longer needed

### 4. Multiple Server Entry Points
- **Issue**: Several files can start the server (railway_startup.py, render_server.py, etc.)
- **Recommended Fix**: Consolidate to one primary method
- **Note**: `backend/app.py` is now the main entry point and properly configured

## What Has Been Fixed

✅ Updated environment.yml to Python 3.11 to match backend requirements
✅ Updated root package.json to have clear, simple scripts for project management
✅ Created BACKEND_CHOICE.md to clarify which backend to use
✅ Removed duplicate vercel.json from backend directory

## Recommended Next Steps

1. **Manual Cleanup**: Remove `rag-chatbot-backend/`, `src/`, and `static/` directories manually
2. **Move Content**: Move `sample_robotics_content.txt` to appropriate location or remove
3. **Test**: Verify backend runs properly with `npm run start-backend`
4. **Deploy**: Deploy `backend/` to Railway and `frontend/rag-chatbot-frontend/` to Vercel with the fixes implemented

## Deployment Commands

For local development:
- Backend: `npm run start-backend` or run `cd backend && python app.py`
- Frontend: `npm run start-frontend` or run `cd frontend/rag-chatbot-frontend && npm start`

The project is now in much better shape with these critical fixes applied!