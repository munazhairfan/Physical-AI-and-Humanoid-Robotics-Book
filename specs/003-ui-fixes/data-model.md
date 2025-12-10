# Data Model: UI Fixes and Improvements

**Date**: 2025-12-10  
**Feature**: UI Fixes and Improvements (`specs/003-ui-fixes`)  
**Status**: Complete  
**Input**: Feature specification from `/specs/003-ui-fixes/spec.md`

## Overview

This document describes the data structures and configuration objects used in the UI fixes and improvements implementation. Since this is primarily a front-end UI enhancement project, the "data model" focuses on configuration objects, CSS custom properties, and UI state structures.

## Theme Configuration Model

### Anime Theme Variables

```typescript
interface AnimeTheme {
  // Primary colors
  primary: string;      // Vibrant red - #ff6b6b
  secondary: string;    // Teal - #4ecdc4  
  accent: string;       // Bright yellow - #ffbe0b
  
  // Background colors
  bgLight: string;      // Light background - #f8f9fa
  bgLightSecondary: string; // Secondary light bg - #ffffff
  bgDark: string;       // Dark background - #1a1a2e
  bgDarkSecondary: string;  // Secondary dark bg - #16213e
  
  // Text colors
  textLight: string;    // Light text - #f1f5f9
  textDark: string;     // Dark text - #1e293b
  textSecondary: string; // Secondary text - #64748b
  
  // Border colors
  borderLight: string;  // Light borders - #e2e8f0
  borderDark: string;   // Dark borders - #334155
  
  // Transition durations
  transitionFast: string;   // 0.2s
  transitionMedium: string; // 0.3s
  transitionSlow: string;   // 0.5s
  
  // Border radius styles
  borderRadiusSharp: string;  // 0px for sharp corners
  borderRadiusSoft: string;   // 8px for soft corners
  borderRadiusRound: string;  // 50% for round elements
  
  // Shadow effects
  shadowLight: string;    // Subtle shadow
  shadowMedium: string;   // Medium shadow
  shadowHeavy: string;    // Heavy shadow
  shadowAnime: string;    // Anime glow effect
}
```

### Robot Theme Fallback Variables

```typescript
interface RobotTheme {
  // Robot-themed colors (fallback)
  primary: string;      // Pink - #ec4899
  secondary: string;    // Yellow/Orange - #f59e0b
  accent: string;       // Red - #f43f5e
  
  // Background colors
  bgLight: string;
  bgLightSecondary: string;
  bgDark: string;
  bgDarkSecondary: string;
  
  // Text and border colors
  textLight: string;
  textDark: string;
  textSecondary: string;
  borderLight: string;
  borderDark: string;
  
  // Standard transitions
  transitionFast: string;
  transitionMedium: string;
  transitionSlow: string;
}
```

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

## Asset Configuration Model

```typescript
interface AssetConfig {
  // SVG icons for homepage features
  homepageIcons: {
    robotBrain: string;      // Path to robot brain SVG
    humanoidRobot: string;   // Path to humanoid robot SVG
    robotLab: string;        // Path to robot lab SVG
    // New anime-themed SVGs
    advancedAI: string;      // Path to advanced AI SVG
    humanoidDesign: string;  // Path to humanoid design SVG
    controlSystems: string;  // Path to control systems SVG
  };
  
  // Static images
  staticImages: {
    chatbot: string;         // Floating chat icon
    favicon: string;         // Site favicon
    logo: string;            // Site logo
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