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
      console.error('Error sending message:', error);

      // Add error message to the chat
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error. Please try again.',
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