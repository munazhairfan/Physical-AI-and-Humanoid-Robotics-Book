import React, { useState, useEffect } from 'react';
import { MainContainer, ChatContainer, MessageList, MessageInput } from '@chatscope/chat-ui-kit-react';
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';
import './chatbot.css';

const ChatbotWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 1,
      message: "Hello! I'm your robotics education assistant. How can I help you with the textbook content today?",
      sender: "bot",
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async (message) => {
    if (!message.trim()) return;

    // Add user message to chat
    const userMessage = {
      id: messages.length + 1,
      message: message,
      sender: "user",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      // Simulate API call to backend
      // In a real implementation, this would call the actual API
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Simulated response based on common robotics questions
      let response = "I can help you with robotics concepts from the textbook. Could you be more specific about what you're looking for?";

      const lowerMsg = message.toLowerCase();
      if (lowerMsg.includes('kinematic') || lowerMsg.includes('forward') || lowerMsg.includes('inverse')) {
        response = "Kinematics is the study of motion without considering the forces that cause it. Forward kinematics calculates end-effector position from joint angles, while inverse kinematics determines joint angles for a desired end-effector position.";
      } else if (lowerMsg.includes('control') || lowerMsg.includes('pid')) {
        response = "Control systems in robotics typically involve feedback mechanisms. PID controllers use Proportional, Integral, and Derivative terms to minimize error: u(t) = K_p e(t) + K_i ∫e(t)dt + K_d de(t)/dt.";
      } else if (lowerMsg.includes('motion') || lowerMsg.includes('path') || lowerMsg.includes('planning')) {
        response = "Motion planning involves finding collision-free paths. Common algorithms include A* for grid-based planning, RRT for sampling-based planning, and optimization-based methods for trajectory generation.";
      } else if (lowerMsg.includes('reinforcement') || lowerMsg.includes('learning') || lowerMsg.includes('rl')) {
        response = "Reinforcement Learning in robotics involves agents learning optimal behaviors through interaction with the environment. Common approaches include Q-Learning, Deep Q-Networks (DQN), and Actor-Critic methods.";
      } else if (lowerMsg.includes('hello') || lowerMsg.includes('hi') || lowerMsg.includes('help')) {
        response = "I'm your robotics education assistant. I can help explain concepts from the textbook including kinematics, control systems, motion planning, perception, and reinforcement learning. What would you like to learn about?";
      }

      // Add bot response to chat
      const botMessage = {
        id: messages.length + 2,
        message: response,
        sender: "bot",
        timestamp: new Date(),
        sources: [],
        confidence: 0.9
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: messages.length + 2,
        message: "Sorry, I encountered an error processing your request. Please try again.",
        sender: "bot",
        timestamp: new Date(),
        sources: [],
        confidence: 0
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (inputValue.trim() && !loading) {
        handleSend(inputValue.trim());
      }
    }
  };

  if (!isOpen) {
    return (
      <div className="chatbot-float-button" onClick={() => setIsOpen(true)}>
        <div className="chatbot-icon">🤖</div>
        <div className="chatbot-label">Robotics Assistant</div>
      </div>
    );
  }

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h3>Robotics Assistant</h3>
        <button className="chatbot-close" onClick={() => setIsOpen(false)}>
          ×
        </button>
      </div>

      <div className="chatbot-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.sender}`}>
            <div className="message-content">
              {msg.message}
              {msg.sources && msg.sources.length > 0 && (
                <div className="message-sources">
                  <details>
                    <summary>Sources ({msg.sources.length})</summary>
                    <ul>
                      {msg.sources.map((source, index) => (
                        <li key={index} className="source-item">
                          <strong>{source.metadata?.module || 'Module'} - {source.metadata?.section || 'Section'}</strong>
                          <p>{source.content?.substring(0, 100)}...</p>
                        </li>
                      ))}
                    </ul>
                  </details>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message bot">
            <div className="message-content">
              <span className="typing-indicator">🤖 Assistant is thinking...</span>
            </div>
          </div>
        )}
      </div>

      <div className="chat-input-container">
        <textarea
          className="chat-input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about robotics concepts..."
          disabled={loading}
          rows="2"
        />
        <button
          className="chat-send-button"
          onClick={() => handleSend(inputValue)}
          disabled={!inputValue.trim() || loading}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatbotWidget;