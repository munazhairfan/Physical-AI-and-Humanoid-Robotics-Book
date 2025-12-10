// Client module to load the floating chat on all pages
import React, { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import FloatingChat from './ChatWidget/FloatingChat';

// This client module will be loaded on all pages by Docusaurus
const FloatingChatLoader = () => {
  useEffect(() => {
    // Create a container div for the floating chat
    const container = document.createElement('div');
    container.id = 'floating-chat-root';
    // Set critical styles to ensure visibility without interfering with page interactions
    container.style.position = 'fixed';
    container.style.zIndex = '2147483647'; // Maximum possible z-index
    container.style.top = '0';
    container.style.left = '0';
    container.style.width = '0';
    container.style.height = '0';
    // The FloatingChat component handles its own positioning and interactions
    // Keep width/height as 0 to avoid interfering with page interactions

    document.body.appendChild(container);

    // Render the FloatingChat component to the container
    const root = createRoot(container);
    root.render(<FloatingChat />);

    // Clean up function
    return () => {
      if (container && container.parentNode) {
        container.parentNode.removeChild(container);
        if (root) {
          root.unmount();
        }
      }
    };
  }, []);

  return null; // This component doesn't render anything itself
};

export default FloatingChatLoader;