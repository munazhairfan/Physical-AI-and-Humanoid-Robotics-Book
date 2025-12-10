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

## Homepage Components API

### Component: HomepageFeatures
- **Purpose**: Displays feature cards with anime-themed icons
- **Props**: None required
- **Structure**: Array of FeatureItem objects
- **FeatureItem**:
  - title: string
  - Svg: React.ComponentType (anime-themed SVG component)
  - description: ReactNode

## Theme Configuration API

### CSS Custom Properties
- **Purpose**: Define anime theme variables
- **Variables**:
  - `--anime-primary-color`: Main theme color
  - `--anime-secondary-color`: Secondary theme color
  - `--anime-accent-color`: Accent color for highlights
  - `--anime-text-color`: Text color for good contrast
  - `--anime-bg-color`: Background color