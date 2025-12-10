# Quickstart Guide: UI Fixes and Improvements

## Setup

1. Ensure you have Node.js >=20.0 installed
2. Navigate to the frontend directory: `cd frontend/rag-chatbot-frontend`
3. Install dependencies: `npm install`
4. Start the development server: `npm start`

## Key Files to Modify

### Chatbot Implementation
- `src/components/FloatingChatLoader.tsx` - Loader for the floating chat
- `src/components/ChatWidget/FloatingChat.tsx` - Main floating chat component
- `src/plugins/floatingChatPlugin.js` - Plugin registration

### Homepage Elements
- `src/pages/index.tsx` - Main homepage with header and buttons
- `src/components/HomepageFeatures/index.tsx` - Feature cards with SVG icons
- `src/components/HomepageFeatures/styles.module.css` - Feature card styling

### Styling
- `src/css/custom.css` - Custom CSS overrides
- `src/pages/index.module.css` - Homepage-specific styles
- Component-specific CSS modules

### SVG Assets
- `ui/` - Directory containing new anime-themed SVGs
- `static/img/` - Current SVG assets

## Development Workflow

1. **Fix Chatbot**: Verify the floating chat is working on all pages
2. **Replace SVGs**: Update homepage cards and feature tabs with anime-themed SVGs
3. **Apply Theme**: Implement anime theme with vibrant colors and sharp shapes
4. **Fix Text Opacity**: Update CSS to improve text visibility
5. **Add Button Links**: Implement navigation for homepage buttons
6. **Remove Unwanted Elements**: Remove keyboard emojis and Docusaurus icons
7. **Test**: Verify all changes work across different pages and screen sizes

## Testing

- Run `npm start` to start the development server
- Navigate to different pages to verify the floating chat appears
- Check that all buttons have proper functionality
- Verify SVG replacements look good and are properly sized
- Confirm text has proper contrast and visibility
- Test responsive design on different screen sizes