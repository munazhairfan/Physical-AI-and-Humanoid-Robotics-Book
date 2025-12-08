// Client module for floating chat that runs on all pages
// This module gets loaded by Docusaurus on every page

import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

if (ExecutionEnvironment.canUseDOM) {
  // Dynamically import and initialize the floating chat when DOM is available
  import('../components/ChatWidget/FloatingChat').then(({ default: FloatingChat }) => {
    // Wait for React to be available
    const checkReact = () => {
      if (window.React && window.ReactDOM) {
        initializeFloatingChat(FloatingChat);
      } else {
        setTimeout(checkReact, 100);
      }
    };

    checkReact();
  }).catch(err => {
    console.error('Failed to load FloatingChat component:', err);
  });
}

function initializeFloatingChat(FloatingChatComponent) {
  // Create a container for the floating chat
  const chatContainer = document.createElement('div');
  chatContainer.id = 'floating-chat-root';
  document.body.appendChild(chatContainer);

  // Use React to render the floating chat component
  const renderFloatingChat = () => {
    const React = window.React;
    const ReactDOM = window.ReactDOM;
    
    if (React && ReactDOM && ReactDOM.createRoot) {
      const root = ReactDOM.createRoot(chatContainer);
      root.render(React.createElement(FloatingChatComponent));
    } else if (React && ReactDOM) {
      // Fallback for older React versions
      ReactDOM.render(React.createElement(FloatingChatComponent), chatContainer);
    }
  };

  renderFloatingChat();
}