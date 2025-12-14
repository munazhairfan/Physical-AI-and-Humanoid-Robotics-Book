# Specification: Text Selection Chatbot Feature

## User Stories

### P1 - Desktop Text Selection
**As a** desktop user
**I want** to select text on any page and see a floating "Ask Chatbot" button appear near my selection
**So that** I can quickly get AI-powered explanations about the selected content

**Acceptance Criteria:**
- When I select at least 10 characters of text, a button labeled "Ask Chatbot" appears near the selection
- The button appears within 100ms of text selection
- The button does not interfere with page interactions
- Clicking the button sends the selected text to the chatbot

### P2 - Mobile Text Selection
**As a** mobile user
**I want** to long-press on text to see a floating "Ask Chatbot" button appear
**So that** I can get AI-powered explanations without complex gestures

**Acceptance Criteria:**
- When I long-press on text for ~600ms, a button labeled "Ask Chatbot" appears near the text
- The button appears without interfering with page scrolling
- Clicking the button sends the detected text to the chatbot

### P3 - Chat Integration
**As a** user who has selected text
**I want** the selected text to be sent to the existing chatbot
**So that** I can get contextually relevant responses about the selected content

**Acceptance Criteria:**
- Selected text is properly captured and sent to the backend
- The chatbot receives the selected text as context
- Response is displayed in the existing chat interface
- The selection button disappears after activation