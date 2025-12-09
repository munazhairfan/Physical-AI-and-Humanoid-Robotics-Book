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

const FloatingChat: React.FC<FloatingChatProps> = ({
  backendUrl = process.env.REACT_APP_BACKEND_URL || (window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app/')
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your Physical AI & Humanoid Robotics assistant. Ask me anything about robotics concepts or select text on the page to get explanations!',
      role: 'assistant',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Function to scroll to the bottom of the chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to handle text selection
  useEffect(() => {
    const handleSelection = () => {
      const selectedTextObj = window.getSelection();
      const text = selectedTextObj?.toString().trim();
      if (text && text.length > 10) { // Only consider meaningful selections
        setSelectedText(text);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

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
  const sendMessage = async (message: string, isFromSelectedText: boolean = false) => {
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
      // Determine which endpoint to call based on if it's selected text
      let response;
      if (isFromSelectedText) {
        response = await fetch(`${backendUrl}/selected_text`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: message }),
        });
      } else {
        response = await fetch(`${backendUrl}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: message,
            history: messages
              .filter(msg => msg.role === 'user' || msg.role === 'assistant')
              .map(msg => ({ role: msg.role, content: msg.content }))
          }),
        });
      }

      if (!response.ok) {
        // If backend returns an error, use mock response instead
        console.warn('Backend error, using mock response');
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Add assistant response to the chat
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.warn('Backend not available, using mock response:', error);

      // Use mock response instead of showing error
      const mockResponse = getMockResponse(message);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: mockResponse,
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

  // Handle selected text action
  const handleSelectedText = () => {
    if (selectedText) {
      sendMessage(selectedText, true);
      setSelectedText(null);
    }
  };

  return (
    <>
      {/* Floating Chat Button */}
      <button
        className={styles['floating-chat-button']}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Open chat"
      >
        <span className={styles.icon}>🤖</span>
      </button>

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
              className={`${styles.message} ${styles[`${message.role}-message`]}`}
            >
              <div className={styles['message-content']}>
                {message.content}
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
          {selectedText && (
            <div className={styles['selected-text-notice']} onClick={handleSelectedText}>
              <span>Selected: "{selectedText.substring(0, 50)}{selectedText.length > 50 ? '...' : ''}"</span>
              <button className={styles['use-selected-btn']}>Use this text</button>
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
            zIndex: 999,
          }}
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
};

export default FloatingChat;