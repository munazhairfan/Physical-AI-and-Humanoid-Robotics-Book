// FloatingChatClientModule.js
// This is a Docusaurus client module that adds the floating chat to all pages

import React, { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import FloatingChat from './ChatWidget/FloatingChat';

// This module runs on the client side after the initial render
export default function FloatingChatClientModule() {
  useEffect(() => {
    // Create a container for the floating chat component
    const container = document.createElement('div');
    container.id = 'floating-chat-root';
    document.body.appendChild(container);

    // Create React root and render the floating chat component
    const root = createRoot(container);
    root.render(<FloatingChat />);

    // Cleanup function to remove the container when component unmounts
    return () => {
      if (container && container.parentNode) {
        container.parentNode.removeChild(container);
      }
      root.unmount();
    };
  }, []);

  // This component doesn't render anything itself
  return null;
}