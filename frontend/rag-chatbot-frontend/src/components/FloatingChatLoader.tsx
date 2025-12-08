// Client module to load the floating chat on all pages
import React from 'react';
import { createRoot } from 'react-dom/client';
import FloatingChat from './ChatWidget/FloatingChat';

// This client module will be loaded on all pages by Docusaurus
function FloatingChatLoader() {
  React.useEffect(() => {
    // Create a container div for the floating chat
    const container = document.createElement('div');
    container.id = 'floating-chat-root';
    document.body.appendChild(container);

    // Render the FloatingChat component to the container
    const root = createRoot(container);
    root.render(<FloatingChat />);

    // Clean up function
    return () => {
      if (container && container.parentNode) {
        container.parentNode.removeChild(container);
        root.unmount();
      }
    };
  }, []);

  return null; // This component doesn't render anything itself
}

export default FloatingChatLoader;