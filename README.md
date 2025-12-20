# Physical AI & Humanoid Robotics Educational Book

This project hosts an educational book on Physical AI and Humanoid Robotics concepts. It features a Docusaurus-based documentation site with multilingual support (English and Urdu), interactive chatbot assistance, and comprehensive content covering robotics, AI, and humanoid systems.

## Features

- 📘 Educational content on Physical AI and Humanoid Robotics
- 🌐 Multilingual support (English and Urdu locales)
- 💬 Interactive chatbot for learning assistance
- 📚 Comprehensive documentation with search functionality
- 🎨 Responsive design with light/dark mode
- 📱 Mobile-friendly interface
- 🔍 Full-text search capabilities
- 📄 Blog system for updates and insights

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Git for version control

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install dependencies

```bash
npm install
```

### 3. Start the development server

```bash
npm run start-frontend
```

The site will be available at http://localhost:3000

### 4. Build for production

```bash
npm run build
```

## Project Structure

- `frontend/rag-chatbot-frontend/` - Docusaurus documentation site
- `backend/` - Backend services (Python-based)
- `static/` - Static assets including chatbot scripts
- `i18n/` - Internationalization files for multiple languages

## Localization

The site supports multiple languages:
- English (default locale)
- Urdu (accessible at `/ur/` paths)

New translations can be added in the `i18n/` directory following Docusaurus i18n conventions.

## Chatbot Integration

An interactive chatbot is integrated throughout the site to assist with learning:
- Appears as a floating chat icon on all pages
- Provides information about robotics concepts
- Responds to text selection for contextual help
- Available in both English and Urdu

## Development

To run the development server:

```bash
npm run start-frontend
```

To build the site for production:

```bash
npm run build
```

## Production Deployment

The site is configured for deployment on Vercel or similar platforms. The build process generates both English and Urdu versions of the site automatically.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For support, please open an issue in the GitHub repository or contact the development team.