# Quickstart: Text Selection Chatbot Feature

## Overview
The text selection chatbot feature allows users to select text on any page and get instant AI-powered explanations by clicking a floating "Ask Chatbot" button that appears near the selection.

## How It Works

### Desktop
1. Select any text on a page by dragging your mouse over it
2. A floating "Ask Chatbot" button will appear near your selection
3. Click the button to send the selected text to the chatbot
4. The chatbot will provide information about the selected text

### Mobile
1. Long-press on any text for ~600ms
2. A floating "Ask Chatbot" button will appear near the text
3. Tap the button to send the text to the chatbot
4. The chatbot will provide information about the selected text

## Keyboard Shortcut
- **Desktop**: `Ctrl/Cmd + Shift + C` after selecting text to activate the chatbot

## Implementation Details

### Files
- `frontend/rag-chatbot-frontend/static/js/selection-chatbot.js` - Core functionality
- `frontend/rag-chatbot-frontend/docusaurus.config.ts` - Script inclusion configuration

### Backend Integration
- Uses existing `/selected_text` endpoint in the backend
- Sends selected text to the backend for AI processing
- Works with existing chatbot infrastructure

### Configuration
- Selection threshold: 10 characters minimum
- Button auto-hide timeout: 3 seconds
- Mobile long-press duration: 600ms

## Troubleshooting

### Button Not Appearing
- Ensure you're selecting at least 10 characters of text
- Check browser console for JavaScript errors
- Verify the script is properly loaded in the page

### Chatbot Not Responding
- Verify backend API is accessible
- Check network tab for API request/response
- Ensure GEMINI_API_KEY is properly configured in environment

## Customization

### Styling
The button appearance can be customized by modifying the CSS in the JavaScript file:
- Background color: Change `#4f46e5` to your preferred color
- Size: Modify padding and font-size values
- Shape: Adjust border-radius for different shapes
- Animation: Adjust transition properties

### Behavior
- Adjust `SELECTION_THRESHOLD` to change minimum text length
- Modify `BUTTON_TIMEOUT` to change auto-hide duration
- Change long-press duration by updating the timeout value