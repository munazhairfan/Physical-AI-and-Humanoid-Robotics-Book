# Data Model: Text Selection Chatbot Feature

## Client-Side State Variables

### Selection Button State
- **selectionButton**: HTMLElement | null
  - Type: DOM Element or null
  - Purpose: Reference to the floating selection button element
  - Lifecycle: Created on first text selection, reused for subsequent selections

### Chatbot State
- **chatbotIframe**: HTMLElement | null
  - Type: iframe element or null
  - Purpose: Reference to the chatbot iframe
  - Lifecycle: Created when needed, removed when chatbot is closed

### Timer State
- **hideTimeout**: number | null
  - Type: Timeout ID or null
  - Purpose: Reference to the timeout that hides the selection button
  - Lifecycle: Set when button appears, cleared when button is used or dismissed

### Visibility State
- **isChatbotVisible**: boolean
  - Type: Boolean
  - Purpose: Tracks whether chatbot is currently visible
  - Values: true/false

## Event Data Structures

### Selection Data
- **selectedText**: string
  - Type: String
  - Purpose: The text that was selected by the user
  - Source: window.getSelection().toString()

### Touch Coordinates
- **startX, startY**: number
  - Type: Number
  - Purpose: Starting coordinates for mobile long-press detection
  - Source: Touch event clientX/clientY properties

## Configuration Constants

### UI Configuration
- **CHATBOT_BUTTON_ID**: string
  - Value: "selection-chatbot-button"
  - Purpose: Unique ID for the selection button element

- **CHATBOT_IFRAME_ID**: string
  - Value: "chatbot-iframe"
  - Purpose: Unique ID for the chatbot iframe element

- **SELECTION_THRESHOLD**: number
  - Value: 10
  - Purpose: Minimum number of characters that must be selected to show button

- **BUTTON_TIMEOUT**: number
  - Value: 3000 (ms)
  - Purpose: Time after which selection button automatically hides

## API Interaction Data

### Message Payload
- **type**: string
  - Value: "SELECTED_TEXT"
  - Purpose: Message type identifier for postMessage communication

- **text**: string
  - Type: String
  - Purpose: The selected text to send to chatbot
  - Source: Selected text from user interaction