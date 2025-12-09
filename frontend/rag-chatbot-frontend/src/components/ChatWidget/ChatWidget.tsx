import React, { useState, useEffect, useRef } from 'react';
import styles from './ChatWidget.module.css';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

const ChatWidget: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: '🤖 Hello! I\'m your Robotics Assistant. Ask me anything about physical AI and humanoid robotics concepts!',
      role: 'assistant',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError(null);

    try {
      // Call backend API - Updated to use configurable URL
      const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || (window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app/');
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          history: messages
            .filter(msg => msg.role === 'user' || msg.role === 'assistant')
            .map(msg => ({
              role: msg.role,
              content: msg.content,
            })),
        }),
      });

      if (!response.ok) {
        // If backend returns an error, use mock response instead
        console.warn('Backend error, using mock response');
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();

      // Add assistant response
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.warn('Backend not available, using mock response:', err);

      // Use mock response instead of showing error
      const mockResponse = getMockResponse(inputValue);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: mockResponse,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.chatHeader}>
        <div className={styles.robotIcon}>🤖</div>
        <h3>Robotics Assistant</h3>
      </div>

      <div className={styles.chatMessages}>
        {messages.map((message) => (
          <div
            key={message.id}
            className={`${styles.message} ${
              message.role === 'user' ? styles.userMessage : styles.assistantMessage
            }`}
          >
            <div className={styles.messageContent}>
              {message.content}
            </div>
            <div className={styles.messageTimestamp}>
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className={styles.message + ' ' + styles.assistantMessage}>
            <div className={styles.messageContent}>
              <div className={styles.typingIndicator}>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className={styles.chatInputForm}>
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
          className={styles.chatButton}
          disabled={isLoading || !inputValue.trim()}
        >
          Send 🚀
        </button>
      </form>

      {error && (
        <div className={styles.error}>
          {error}
        </div>
      )}
    </div>
  );
};

export default ChatWidget;