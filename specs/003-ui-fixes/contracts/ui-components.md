# UI Component Contracts

## Floating Chat Component API

### Component: FloatingChat
- **Purpose**: Provides a floating chat interface accessible from all pages
- **Props**: None required
- **State**:
  - isOpen: boolean (whether chat is open/closed)
  - isVisible: boolean (whether chat button is visible)
- **Events**:
  - onToggle: triggered when chat is opened/closed
  - onMessage: triggered when user sends a message

### Component: FloatingChatLoader
- **Purpose**: Loads the floating chat component on all pages
- **Props**: None required
- **Behavior**: Mounts FloatingChat component to DOM on all pages
