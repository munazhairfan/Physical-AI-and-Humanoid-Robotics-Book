import React, { useState, useEffect, useRef } from 'react';
import styles from './FloatingChat.module.css';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

interface FloatingChatProps {
  backendUrl?: string;
}

const FloatingChat: React.FC<FloatingChatProps> = ({ backendUrl }) => {
  // Initialize backend URL inside the component to avoid process issues
  const [resolvedBackendUrl, setResolvedBackendUrl] = useState(() => {
    // Server-side rendering check - window is not available during SSR
    if (typeof window === 'undefined') {
      // Server-side fallback
      return 'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app/';
    }

    // Client-side - window is available
    // Get the backend URL from a global variable that can be set during deployment
    // This avoids the process.env issue in browser environments
    const envBackendUrl = (window as any).REACT_APP_BACKEND_URL ||
                         (window as any).env?.REACT_APP_BACKEND_URL;

    // Check if we're in browser environment and use appropriate URL
    return window.location.hostname === 'localhost'
      ? 'http://localhost:8000'
      : (envBackendUrl || 'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app/');
  });

  // Use the provided backendUrl if available, otherwise use the resolved one
  const currentBackendUrl = backendUrl || resolvedBackendUrl;

  // Debug log to see if component is rendering
  console.log('FloatingChat component rendered');
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your Physical AI & Humanoid Robotics assistant. Ask me anything about robotics concepts!',
      role: 'assistant',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTextSelectMenuOpen, setIsTextSelectMenuOpen] = useState(false);
  const [textSelectPosition, setTextSelectPosition] = useState({ x: 0, y: 0 });
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Function to scroll to the bottom of the chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Add a resize handler for better responsiveness on small devices
  useEffect(() => {
    const handleResize = () => {
      // Force re-render to adjust to new viewport dimensions
      setIsOpen(prev => prev); // This will trigger a re-render without changing state
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Text selection functionality
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      if (selection && selection.toString().trim() !== '') {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();

        // Show the context menu near the selected text
        setTextSelectPosition({
          x: rect.left + window.scrollX,
          y: rect.top + window.scrollY - 40 // Position above the selection
        });
        setSelectedText(selection.toString().trim());
        setIsTextSelectMenuOpen(true);
      } else {
        setIsTextSelectMenuOpen(false);
      }
    };

    const handleMouseDown = (e: MouseEvent) => {
      // Close the context menu if clicking outside of it
      if (isTextSelectMenuOpen && !(e.target as Element).closest('.text-select-menu')) {
        setIsTextSelectMenuOpen(false);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('mousedown', handleMouseDown);

    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('mousedown', handleMouseDown);
    };
  }, [isTextSelectMenuOpen]);

  // Mock response function for when backend is not available
  const getMockResponse = (userMessage: string): string => {
    const lowerCaseMsg = userMessage.toLowerCase();

    if (lowerCaseMsg.includes('hello') || lowerCaseMsg.includes('hi') || lowerCaseMsg.includes('hey')) {
      return 'Hello there! I\'m your Robotics Assistant. How can I help you with robotics concepts today?';
    } else if (lowerCaseMsg.includes('robot') || lowerCaseMsg.includes('ai')) {
      return 'Robots and AI are fascinating fields! Physical AI involves creating intelligent systems that interact with the physical world. Humanoid robots specifically mimic human form and behavior. What specific aspect would you like to know more about?';
    } else if (lowerCaseMsg.includes('learn') || lowerCaseMsg.includes('study') || lowerCaseMsg.includes('education')) {
      return 'Learning about robotics involves understanding mechanics, electronics, programming, and artificial intelligence. Key areas include kinematics, control systems, sensors, actuators, and machine learning. Would you like to dive deeper into any specific area?';
    } else if (lowerCaseMsg.includes('humanoid') || lowerCaseMsg.includes('bipedal') || lowerCaseMsg.includes('walking')) {
      return 'Humanoid robots are designed to resemble and mimic human behavior. Key challenges include balance control, gait planning, and natural movement. They often use inverse kinematics and PID controllers for smooth motion. What would you like to know about humanoid robotics?';
    } else if (lowerCaseMsg.includes('control') || lowerCaseMsg.includes('motion') || lowerCaseMsg.includes('movement')) {
      return 'Robot control involves various techniques like PID controllers, inverse kinematics, and trajectory planning. For humanoid robots, maintaining balance and creating natural movements require sophisticated control algorithms. Would you like to know about a specific control method?';
    } else {
      const responses = [
        'That\'s an interesting question about robotics! Physical AI and humanoid robotics involve complex interactions between mechanical systems, sensors, and artificial intelligence. Could you elaborate on what specifically interests you?',
        'Robots are amazing systems that combine mechanics, electronics, and software. In humanoid robotics, we focus on creating machines that can interact with humans and environments in human-like ways. What aspect would you like to explore?',
        'Great question! Robotics encompasses many fields including kinematics, dynamics, control systems, and AI. Physical AI specifically deals with robots that interact with the physical world. What specific area interests you most?',
        'I appreciate your interest in robotics technology! There are many fascinating aspects to explore, from the mechanics of robot movement to the AI that powers their decision-making. What would you like to know more about?',
        'Robotics is an interdisciplinary field combining engineering and computer science. Humanoid robots add the complexity of mimicking human form and function. What specific topic would you like to discuss?'
      ];
      return responses[Math.floor(Math.random() * responses.length)];
    }
  };

  // Function to send a message to the backend
  const sendMessage = async (message: string) => {
    if (!message.trim() || isLoading) return;

    // Add user message to the chat
    const userMessage: Message = {
      id: Date.now().toString(),
      content: message,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Ensure there's no trailing slash in the backend URL to prevent double slashes
      const normalizedBackendUrl = currentBackendUrl.replace(/\/$/, '');
      console.log('Attempting to connect to backend:', `${normalizedBackendUrl}/chat`);
      console.log('Sending message:', message);

      // Try the configured backend URL
      const response = await fetch(`${normalizedBackendUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          history: messages
            .filter(msg => msg.role === 'user' || msg.role === 'assistant')
            .map(msg => ({ role: msg.role, content: msg.content }))
        }),
      });

      console.log('Response received:', response.status, response.ok);

      if (!response.ok) {
        // If backend returns an error, use mock response instead
        console.warn('Backend error, using mock response');
        console.warn('Response status:', response.status);
        console.warn('Response status text:', response.statusText);
        const errorText = await response.text();
        console.warn('Error details:', errorText);
        throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
      }

      const data = await response.json();
      console.log('Backend response data:', data);

      // Add assistant response to the chat
      // Check if response has the expected structure
      let responseContent = '';
      if (data && typeof data === 'object') {
        responseContent = data.response || data.text || data.message || '';
      } else {
        responseContent = String(data || '');
      }

      // Ensure content is a proper string
      responseContent = typeof responseContent === 'string' ? responseContent : JSON.stringify(responseContent) || '';

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: responseContent,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Full error object:', error);
      console.warn('Backend not available, using mock response:', error);

      // Provide a more informative error message to the user
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `I'm sorry, I'm having trouble connecting to the backend service. Using mock responses. Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to send selected text to the backend
  const sendSelectedText = async () => {
    if (!selectedText.trim() || isLoading) return;

    // Add user message (selected text) to the chat
    const userMessage: Message = {
      id: Date.now().toString(),
      content: `About this text: "${selectedText.substring(0, 100)}${selectedText.length > 100 ? '...' : ''}"`,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setIsTextSelectMenuOpen(false); // Close the context menu

    try {
      // Ensure there's no trailing slash in the backend URL to prevent double slashes
      const normalizedBackendUrl = currentBackendUrl.replace(/\/$/, '');
      console.log('Attempting to connect to backend:', `${normalizedBackendUrl}/selected_text`);
      console.log('Sending selected text:', selectedText);

      // Try the configured backend URL with the selected_text endpoint
      const response = await fetch(`${normalizedBackendUrl}/selected_text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          text: selectedText,
          context: `User selected this text: "${selectedText}" and asked for more information.`
        }),
      });

      console.log('Response received:', response.status, response.ok);

      if (!response.ok) {
        // If backend returns an error, use mock response instead
        console.warn('Backend error for selected text, using mock response');
        console.warn('Response status:', response.status);
        console.warn('Response status text:', response.statusText);
        const errorText = await response.text();
        console.warn('Error details:', errorText);
        throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
      }

      const data = await response.json();
      console.log('Backend response data for selected text:', data);

      // Add assistant response to the chat
      // Check if response has the expected structure
      let responseContent = '';
      if (data && typeof data === 'object') {
        responseContent = data.response || data.text || data.message || '';
      } else {
        responseContent = String(data || '');
      }

      // Ensure content is a proper string
      responseContent = typeof responseContent === 'string' ? responseContent : JSON.stringify(responseContent) || '';

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: responseContent,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Full error object for selected text:', error);
      console.warn('Selected text backend not available, using mock response:', error);

      // Provide a more informative error message to the user
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `I'm sorry, I'm having trouble processing the selected text. Using mock response. Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputValue);
  };

  return (
    <div>
      {/* Floating Chat Button */}
      <button
        className={styles['floating-chat-button']}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Open chat"
      >
        <img
          src="/img/chatbot.svg"
          alt="Chat"
          className={styles.chatIcon}
          style={{ width: '70%', height: '70%', objectFit: 'contain' }}
        />
      </button>

      {/* Text Selection Context Menu */}
      {isTextSelectMenuOpen && (
        <div
          className={styles['text-select-menu']}
          style={{
            left: textSelectPosition.x,
            top: textSelectPosition.y,
          }}
          onClick={sendSelectedText}
        >
          Ask Chatbot
        </div>
      )}

      {/* Chat Popup */}
      <div className={`${styles['chat-popup']} ${isOpen ? styles.open : ''}`}>
        <div className={styles['chat-popup-header']}>
          <h3>Robotics Assistant</h3>
          <button
            className={styles['close-btn']}
            onClick={() => setIsOpen(false)}
            aria-label="Close chat"
          >
            ×
          </button>
        </div>

        <div className={styles['chat-messages']}>
          {messages.map((message) => (
            <div
              key={message.id}
              className={`${styles.message} ${message.role === 'user' ? styles['user-message'] : message.role === 'assistant' ? styles['assistant-message'] : styles['assistant-message']}`}
            >
              <div className={styles['message-content']}>
                {String(message.content || '')}
              </div>
              <div className={styles['message-timestamp']}>
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className={`${styles.message} ${styles['assistant-message']}`}>
              <div className={styles['message-content']}>
                <div className={styles['typing-indicator']}>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form className={styles['chat-input-form']} onSubmit={handleSubmit}>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask about robotics concepts..."
            className={styles.chatInput}
            disabled={isLoading}
          />
          <button
            type="submit"
            className={styles['send-button']}
            disabled={isLoading || !inputValue.trim()}
          >
            Send
          </button>
        </form>
      </div>

      {/* Overlay when chat is open */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.2)',
            zIndex: 1040, /* Bootstrap modal backdrop level */
          }}
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  );
};

export default FloatingChat;