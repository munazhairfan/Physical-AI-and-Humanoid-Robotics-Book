import React, { useState, useEffect, useRef } from 'react';
import './Chatbot.css'; // We'll create this CSS file as well

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

interface ChatbotProps {
  backendUrl?: string; // URL of the FastAPI backend
}

const Chatbot: React.FC<ChatbotProps> = ({ backendUrl = 'http://localhost:8000' }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your RAG Chatbot. How can I help you today?',
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
      if (text) {
        setSelectedText(text);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  // Function to send a message to the backend
  const sendMessage = async (message: string, isSelectedText: boolean = false) => {
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
      if (isSelectedText) {
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
      setSelectedText(null); // Clear the selected text after sending
    }
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h3>RAG Chatbot</h3>
        {selectedText && (
          <div className="selected-text-notice" onClick={handleSelectedText}>
            <span>Selected: "{selectedText.substring(0, 50)}{selectedText.length > 50 ? '...' : ''}"</span>
            <button className="use-selected-btn">Use this text</button>
          </div>
        )}
      </div>

      <div className="chatbot-messages">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
          >
            <div className="message-content">
              {message.content}
            </div>
            <div className="message-timestamp">
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message assistant-message">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chatbot-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type your message here..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !inputValue.trim()}>
          Send
        </button>
      </form>
    </div>
  );
};

export default Chatbot;