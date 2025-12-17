# Data Model: UI Fixes and Improvements

**Date**: 2025-12-10  
**Feature**: UI Fixes and Improvements (`specs/003-ui-fixes`)  
**Status**: Complete  
**Input**: Feature specification from `/specs/003-ui-fixes/spec.md`

## Overview

This document describes the data structures and configuration objects used in the UI fixes and improvements implementation. Since this is primarily a front-end UI enhancement project, the "data model" focuses on configuration objects, CSS custom properties, and UI state structures.


## UI Component State Models

### Floating Chat Component

```typescript
interface FloatingChatState {
  isOpen: boolean;              // Visibility state of chat popup
  messages: Message[];          // Array of chat messages
  inputValue: string;           // Current input text
  isLoading: boolean;           // Loading state for API calls
  resolvedBackendUrl: string;   // Resolved backend URL
  currentBackendUrl: string;    // Active backend URL being used
}

interface Message {
  id: string;           // Unique message identifier
  content: string;      // Message text content
  role: 'user' | 'assistant'; // Message sender role
  timestamp: Date;      // When the message was created
}
```

### Homepage Header State

```typescript
interface HomepageHeaderState {
  siteConfig: SiteConfig;   // Docusaurus site configuration
  currentTheme: 'light' | 'dark'; // Current theme mode
}

interface SiteConfig {
  title: string;          // Site title
  tagline: string;        // Site tagline
  favicon: string;        // Favicon path
  url: string;            // Production URL
  baseUrl: string;        // Base path
}
```

## CSS Custom Properties Model

The theme is implemented using CSS custom properties (CSS variables) that can be dynamically updated:

### Light Theme Properties
```css
:root {
  --background-primary: #ffffff;
  --background-secondary: #f8fafc;
  --text-primary: #1e293b;
  --text-secondary: #64748b;
  --accent-primary: #8b5cf6;
  --accent-secondary: #06b6d4;
  --border-color: #e2e8f0;
}
```

### Dark Theme Properties
```css
html[data-theme='dark'] {
  --background-primary: #0f172a;
  --background-secondary: #1e293b;
  --text-primary: #f1f5f9;
  --text-secondary: #cbd5e1;
  --accent-primary: #a78bfa;
  --accent-secondary: #22d3ee;
  --border-color: #334155;
}
```

## Animation Configuration Model

```typescript
interface AnimationConfig {
  // Pulsing animation properties
  pulse: {
    keyframes: string[];
    duration: string;
    iterations: string;
  };
  
  // Floating animation properties
  float: {
    keyframes: string[];
    duration: string;
    easing: string;
  };
  
  // Transition properties
  transition: {
    fast: string;
    medium: string;
    slow: string;
  };
}
```


## UI State Flow Model

```typescript
interface UIStateFlow {
  // Initial state when page loads
  initial: {
    chatOpen: false;
    currentTheme: 'auto'; // Based on system preference
    animationsEnabled: true;
  };
  
  // State transitions
  transitions: {
    // When chat button is clicked
    chatButtonClick: {
      from: { chatOpen: false };
      to: { chatOpen: true };
      action: 'animateChatOpen';
    };
    
    // When theme is toggled
    themeToggle: {
      from: { currentTheme: 'light' | 'dark' };
      to: { currentTheme: 'dark' | 'light' };
      action: 'updateCSSVariables';
    };
  };
}
```

## Responsive Design Configuration

```typescript
interface ResponsiveConfig {
  breakpoints: {
    mobile: string;      // 768px
    tablet: string;      // 996px
    desktop: string;     // 1200px
  };
  
  // Component-specific responsive rules
  components: {
    chatWidget: {
      mobile: {
        width: string;   // calc(100vw - 20px)
        height: string;  // 60vh
        right: string;   // 10px
        left: string;    // 10px
      };
      desktop: {
        width: string;   // 420px max
        height: string;  // 60vh
        right: string;   // 25px
        left: string;    // auto
      };
    };
  };
}
```

## Data Flow and Interactions

The UI fixes and improvements implement a reactive data flow where:

1. **Theme state** is managed through CSS custom properties
2. **Component states** are managed through React useState/useEffect
3. **Global states** (like chat visibility) use browser storage for persistence
4. **Animation states** are triggered by component state changes
5. **Responsive states** adapt based on viewport dimensions

This approach ensures consistent theming while maintaining performance and accessibility.