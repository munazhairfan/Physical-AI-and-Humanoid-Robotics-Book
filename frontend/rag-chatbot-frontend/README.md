# Physical AI & Humanoid Robotics - Frontend

This is the Docusaurus-based frontend for the Physical AI & Humanoid Robotics RAG chatbot.

## Features

- Interactive chat interface for asking robotics questions
- Real-time communication with the backend API
- Responsive design that works on desktop and mobile
- Clean, modern UI with typing indicators and message history

## Installation

```bash
npm install
```

## Local Development

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Building for Production

```bash
npm run build
```

This generates static files in the `build/` directory.

## Deployment

### GitHub Pages

The site is configured for GitHub Pages deployment:

1. The `docusaurus.config.ts` is configured with:
   - `baseUrl: '/Physical-AI-and-Humanoid-Robotics/'`
   - `organizationName: 'your-username'`
   - `projectName: 'Physical-AI-and-Humanoid-Robotics'`

2. To deploy:
   - Push your changes to the `main` branch
   - Go to repository Settings → Pages
   - Set source to "Deploy from a branch"
   - Select branch `main`, folder `/docs`

### API Configuration

The frontend connects to the backend API at `http://localhost:8000` during development. For production deployment, you may need to update the API endpoint in `src/components/ChatWidget/ChatWidget.tsx` to point to your deployed backend.

## Integration with Backend

The frontend communicates with the backend API using these endpoints:
- `POST /chat` - For chat conversations
- `POST /book_query` - For book/document queries
- `POST /selected_text` - For selected text processing

## Directory Structure

- `src/pages/index.tsx` - Main page with chat interface
- `src/components/ChatWidget/` - Chat interface component
- `docs/` - Documentation pages
- `static/` - Static assets like images
