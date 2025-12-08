# Backend Directory Choice

## Important Notice

This project contains two backend directories:
- `backend/` - **ACTIVE AND MAINTAINED** - Use this for deployment
- `rag-chatbot-backend/` - **DUPLICATE/OUTDATED** - Do not use

## Current Active Backend

The project uses `backend/` as the main backend directory. This has been configured and tested for Railway deployment with the fixes needed to resolve the import errors.

## For Deployment

- Use `backend/` directory for backend deployment
- Frontend connects to backend via configured API endpoints
- Do not use `rag-chatbot-backend/` directory

## Directory Cleanup

Once deployment is working properly, the redundant `rag-chatbot-backend/` directory may be removed.