---
title: "RAG Chatbot: UI Integration"
description: "User interface integration for RAG-based chatbot in robotics education"
sidebar_position: 3
slug: /rag-chatbot/ui-integration
keywords: [RAG, chatbot, UI, React, Docusaurus, robotics, education]
---

# RAG Chatbot: UI Integration

## Overview

This section covers the integration of the RAG-based chatbot into the Docusaurus documentation interface for robotics education. The integration provides students with an AI-powered assistant that can answer questions about the textbook content directly within the documentation pages.

## Architecture

### Frontend Tech Stack

The chatbot UI is built using modern web technologies:

```javascript
// package.json dependencies
{
  "dependencies": {
    "@docusaurus/core": "^3.0.0",
    "@docusaurus/module-type-aliases": "^3.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "@chatscope/chat-ui-kit-react": "^2.0.3",
    "@chatscope/chat-ui-kit-styles": "^1.2.0",
    "axios": "^1.6.0",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.1"
  }
}
```

### Component Structure

```
src/
├── components/
│   ├── Chatbot/
│   │   ├── ChatbotWidget.jsx
│   │   ├── ChatWindow.jsx
│   │   ├── Message.jsx
│   │   ├── InputArea.jsx
│   │   └── ChatHistory.jsx
│   └── RAG/
│       ├── RAGProvider.jsx
│       └── useRAGQuery.js
```

## Core Components

### 1. Main Chatbot Widget

```jsx
// src/components/Chatbot/ChatbotWidget.jsx
import React, { useState, useEffect } from 'react';
import { MainContainer, ChatContainer, MessageList, Message, MessageInput } from '@chatscope/chat-ui-kit-react';
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';
import { useRAGQuery } from '../RAG/useRAGQuery';
import { ChatWindow } from './ChatWindow';
import { InputArea } from './InputArea';

const ChatbotWidget = ({ pageContext }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 1,
      message: "Hello! I'm your robotics education assistant. How can I help you with the textbook content today?",
      sender: "bot",
      timestamp: new Date()
    }
  ]);

  const { query, loading, error, response } = useRAGQuery();

  const handleSend = async (message) => {
    // Add user message to chat
    const userMessage = {
      id: messages.length + 1,
      message: message,
      sender: "user",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);

    // Query the RAG system
    const ragResponse = await query({
      query: message,
      context: pageContext, // Current page context for relevance
      max_results: 5
    });

    // Add bot response to chat
    const botMessage = {
      id: messages.length + 2,
      message: ragResponse?.response || "I encountered an issue processing your query. Please try again.",
      sender: "bot",
      timestamp: new Date(),
      sources: ragResponse?.sources || [],
      confidence: ragResponse?.confidence || 0
    };

    setMessages(prev => [...prev, botMessage]);
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
      <ChatWindow messages={messages} loading={loading} />
      <InputArea onSend={handleSend} disabled={loading} />
    </div>
  );
};

export default ChatbotWidget;
```

### 2. Chat Window Component

```jsx
// src/components/Chatbot/ChatWindow.jsx
import React from 'react';
import { ChatContainer, MessageList, Message } from '@chatscope/chat-ui-kit-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { MessageComponent } from './Message';

export const ChatWindow = ({ messages, loading }) => {
  return (
    <ChatContainer>
      <MessageList>
        {messages.map((msg) => (
          <MessageComponent
            key={msg.id}
            message={msg.message}
            sender={msg.sender}
            timestamp={msg.timestamp}
            sources={msg.sources}
            confidence={msg.confidence}
          />
        ))}
        {loading && (
          <div className="typing-indicator">
            <span>🤖 Assistant is thinking...</span>
          </div>
        )}
      </MessageList>
    </ChatContainer>
  );
};
```

### 3. Message Component with Source Attribution

```jsx
// src/components/Chatbot/Message.jsx
import React from 'react';
import { Message as ChatMessage } from '@chatscope/chat-ui-kit-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export const MessageComponent = ({ message, sender, timestamp, sources, confidence }) => {
  const isBot = sender === 'bot';

  return (
    <ChatMessage
      model={{
        message: message,
        sender: sender,
        direction: isBot ? 'incoming' : 'outgoing',
        position: 'normal'
      }}
    >
      <ChatMessage.Message>
        <ReactMarkdown remarkPlugins={[remarkGfm]} className="chat-message-content">
          {message}
        </ReactMarkdown>

        {isBot && sources && sources.length > 0 && (
          <div className="message-sources">
            <details>
              <summary>Sources ({sources.length})</summary>
              <ul>
                {sources.map((source, index) => (
                  <li key={index} className="source-item">
                    <strong>{source.metadata.module || 'Module'} - {source.metadata.section || 'Section'}</strong>
                    <p>{source.content.substring(0, 100)}...</p>
                    <small>Relevance: {(source.relevance_score * 100).toFixed(1)}%</small>
                  </li>
                ))}
              </ul>
            </details>
          </div>
        )}

        {isBot && confidence !== undefined && (
          <div className="confidence-indicator">
            Confidence: {Math.round(confidence * 100)}%
          </div>
        )}
      </ChatMessage.Message>
    </ChatMessage>
  );
};
```

### 4. Input Area with Suggestions

```jsx
// src/components/Chatbot/InputArea.jsx
import React, { useState } from 'react';
import { MessageInput } from '@chatscope/chat-ui-kit-react';

export const InputArea = ({ onSend, disabled }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSend = (value) => {
    if (value.trim() && !disabled) {
      onSend(value.trim());
      setInputValue('');
    }
  };

  // Suggested questions based on robotics education
  const suggestedQuestions = [
    "Explain forward kinematics",
    "How does inverse kinematics work?",
    "What is a PID controller?",
    "Describe the RRT algorithm",
    "How do Kalman filters work in robotics?",
    "What is SLAM?"
  ];

  return (
    <div className="chat-input-container">
      <MessageInput
        placeholder="Ask about robotics concepts..."
        value={inputValue}
        onChange={(val) => setInputValue(val)}
        onSend={handleSend}
        disabled={disabled}
        attachButton={false}
      />

      {!disabled && inputValue === '' && (
        <div className="suggested-questions">
          <small>Try asking:</small>
          <div className="question-tags">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                className="question-tag"
                onClick={() => onSend(question)}
                disabled={disabled}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
```

## RAG Query Hook

```javascript
// src/components/RAG/useRAGQuery.js
import { useState } from 'react';
import axios from 'axios';

export const useRAGQuery = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);

  const query = async (params) => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('/api/query', params, {
        headers: {
          'Content-Type': 'application/json',
        }
      });

      setResponse(response.data);
      return response.data;
    } catch (err) {
      setError(err.message || 'Error querying the RAG system');
      console.error('RAG Query Error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { query, loading, error, response };
};
```

## Docusaurus Integration

### Creating the Chatbot Plugin

```javascript
// src/plugins/chatbot/index.js
const path = require('path');

module.exports = function (context, options) {
  return {
    name: 'docusaurus-plugin-chatbot',

    getPathsToWatch() {
      return [path.join(__dirname, '../components/Chatbot/**/*.{js,jsx,ts,tsx}')];
    },

    configureWebpack(config, isServer, utils) {
      return {
        resolve: {
          alias: {
            '@chatbot': path.resolve(__dirname, '../components/Chatbot'),
          },
        },
      };
    },

    injectHtmlTags() {
      return {
        postBodyTags: [
          `<div id="chatbot-root"></div>`,
        ],
      };
    },
  };
};
```

### Layout Wrapper Component

```jsx
// src/theme/Layout/index.jsx
import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatbotWidget from '@chatbot/ChatbotWidget';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props} />
      <ChatbotWidget pageContext={props} />
    </>
  );
}
```

### Page Context Provider

```jsx
// src/components/RAG/RAGProvider.jsx
import React, { createContext, useContext, useState } from 'react';

const RAGContext = createContext();

export const RAGProvider = ({ children, pageMetadata }) => {
  const [currentContext, setCurrentContext] = useState({
    page: pageMetadata,
    section: null,
    module: null
  });

  return (
    <RAGContext.Provider value={{ currentContext, setCurrentContext }}>
      {children}
    </RAGContext.Provider>
  );
};

export const useRAGContext = () => {
  const context = useContext(RAGContext);
  if (!context) {
    throw new Error('useRAGContext must be used within a RAGProvider');
  }
  return context;
};
```

## Styling

### CSS Styles

```css
/* src/css/chatbot.css */
.chatbot-container {
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 400px;
  height: 600px;
  background: white;
  border: 1px solid #ddd;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  display: flex;
  flex-direction: column;
  z-index: 1000;
}

.chatbot-header {
  background: #2563eb;
  color: white;
  padding: 12px 16px;
  border-radius: 12px 12px 0 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chatbot-close {
  background: none;
  border: none;
  color: white;
  font-size: 20px;
  cursor: pointer;
}

.chatbot-float-button {
  position: fixed;
  bottom: 20px;
  right: 20px;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  transition: all 0.3s ease;
}

.chatbot-float-button:hover {
  transform: scale(1.05);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
}

.chatbot-icon {
  font-size: 24px;
  margin-bottom: 4px;
}

.chatbot-label {
  font-size: 12px;
  font-weight: bold;
}

.chat-message-content {
  line-height: 1.5;
}

.chat-message-content code {
  background: #f1f5f9;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: monospace;
}

.chat-message-content pre {
  background: #f8fafc;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
  margin: 8px 0;
}

.message-sources {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #e2e8f0;
}

.source-item {
  margin: 8px 0;
  padding: 8px;
  background: #f8fafc;
  border-radius: 4px;
  font-size: 12px;
}

.source-item p {
  margin: 4px 0;
  color: #64748b;
}

.confidence-indicator {
  font-size: 11px;
  color: #64748b;
  margin-top: 4px;
}

.suggested-questions {
  padding: 12px;
  border-top: 1px solid #e2e8f0;
}

.question-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.question-tag {
  background: #e0f2fe;
  border: 1px solid #7dd3fc;
  border-radius: 20px;
  padding: 6px 12px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
}

.question-tag:hover {
  background: #bae6fd;
  transform: translateY(-1px);
}

.typing-indicator {
  padding: 12px;
  color: #64748b;
  font-style: italic;
}

@media (max-width: 768px) {
  .chatbot-container {
    width: calc(100% - 40px);
    height: 50vh;
    bottom: 10px;
    right: 10px;
  }

  .chatbot-float-button {
    width: 50px;
    height: 50px;
    bottom: 15px;
    right: 15px;
  }
}
```

## Advanced Features

### Context-Aware Responses

```jsx
// Enhanced chatbot widget with context awareness
import React, { useState, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';

const ContextAwareChatbot = () => {
  const location = useLocation();
  const [pageContext, setPageContext] = useState({});

  useEffect(() => {
    // Extract context from current page
    const pathParts = location.pathname.split('/').filter(Boolean);
    if (pathParts.length >= 2) {
      setPageContext({
        module: pathParts[1], // e.g., 'module-1'
        section: pathParts[2] || '', // e.g., 'overview'
        url: location.pathname
      });
    }
  }, [location.pathname]);

  return (
    <ChatbotWidget pageContext={pageContext} />
  );
};

export default ContextAwareChatbot;
```

### Conversation History Persistence

```javascript
// src/components/Chatbot/useChatHistory.js
import { useState, useEffect } from 'react';

const CHAT_HISTORY_KEY = 'robotics-chat-history';

export const useChatHistory = (conversationId) => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const savedHistory = localStorage.getItem(`${CHAT_HISTORY_KEY}-${conversationId}`);
    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error('Error loading chat history:', e);
      }
    }
  }, [conversationId]);

  const saveHistory = (newHistory) => {
    setHistory(newHistory);
    localStorage.setItem(`${CHAT_HISTORY_KEY}-${conversationId}`, JSON.stringify(newHistory));
  };

  const clearHistory = () => {
    localStorage.removeItem(`${CHAT_HISTORY_KEY}-${conversationId}`);
    setHistory([]);
  };

  return { history, saveHistory, clearHistory };
};
```

### Smart Suggestions Based on Current Page

```jsx
// src/components/Chatbot/SmartSuggestions.jsx
import React from 'react';

const SmartSuggestions = ({ currentPage, onSuggestionClick }) => {
  // Define suggestions based on current page/module
  const getSuggestions = () => {
    if (currentPage.includes('kinematics')) {
      return [
        "Explain forward kinematics with an example",
        "How do you solve inverse kinematics for a 2-DOF arm?",
        "What are Jacobians used for?",
        "Show me the DH parameters convention"
      ];
    } else if (currentPage.includes('control')) {
      return [
        "How does a PID controller work?",
        "What are the tuning methods for PID?",
        "Explain Model Predictive Control",
        "How do you design a stable controller?"
      ];
    } else if (currentPage.includes('planning')) {
      return [
        "How does the A* algorithm work?",
        "What is the configuration space?",
        "Explain RRT path planning",
        "How do you handle dynamic obstacles?"
      ];
    } else if (currentPage.includes('perception')) {
      return [
        "How does SLAM work?",
        "Explain Kalman filters",
        "What is sensor fusion?",
        "How do you calibrate sensors?"
      ];
    }

    // Default suggestions
    return [
      "What is covered in this section?",
      "Explain the key concepts",
      "Show me an example",
      "How does this apply to real robots?"
    ];
  };

  const suggestions = getSuggestions();

  return (
    <div className="smart-suggestions">
      <small>Related questions:</small>
      <div className="suggestion-tags">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            className="suggestion-tag"
            onClick={() => onSuggestionClick(suggestion)}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
};

export default SmartSuggestions;
```

## Accessibility Features

### Keyboard Navigation

```jsx
// Enhanced input area with keyboard shortcuts
import React, { useState, useRef, useEffect } from 'react';

export const AccessibleInputArea = ({ onSend, disabled }) => {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef(null);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (inputValue.trim() && !disabled) {
        onSend(inputValue.trim());
        setInputValue('');
      }
    }
  };

  return (
    <div className="chat-input-container" role="form" aria-label="Chat input">
      <textarea
        ref={inputRef}
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about robotics concepts..."
        disabled={disabled}
        aria-label="Type your question about robotics"
        rows="2"
        className="accessible-textarea"
      />
      <button
        onClick={() => {
          if (inputValue.trim() && !disabled) {
            onSend(inputValue.trim());
            setInputValue('');
          }
        }}
        disabled={disabled || !inputValue.trim()}
        aria-label="Send message"
      >
        Send
      </button>
    </div>
  );
};
```

## Performance Optimization

### Lazy Loading

```jsx
// Lazy load chatbot component
import React, { lazy, Suspense } from 'react';

const LazyChatbot = lazy(() => import('./ChatbotWidget'));

const ChatbotWithSuspense = () => (
  <Suspense fallback={<div className="chatbot-loading">Loading assistant...</div>}>
    <LazyChatbot />
  </Suspense>
);

export default ChatbotWithSuspense;
```

## Deployment Configuration

### Environment Configuration

```javascript
// docusaurus.config.js - Add to the config
module.exports = {
  // ... other config
  plugins: [
    // ... other plugins
    './src/plugins/chatbot', // Your chatbot plugin
  ],

  themeConfig: {
    // ... other theme config
    chatbot: {
      enabled: true,
      apiEndpoint: process.env.RAG_API_ENDPOINT || '/api',
      defaultOpen: false,
      showOnAllPages: true,
      // Page-specific configurations
      pageConfig: {
        includeInPages: ['docs/**/*'], // Only show on docs pages
        excludeFromPages: ['blog/**/*'], // Don't show on blog pages
      }
    }
  },
};
```

## Testing

### Component Testing Example

```javascript
// tests/Chatbot.test.jsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ChatbotWidget from '../src/components/Chatbot/ChatbotWidget';

jest.mock('../src/components/RAG/useRAGQuery', () => ({
  useRAGQuery: () => ({
    query: jest.fn().mockResolvedValue({
      response: 'Test response',
      sources: [],
      confidence: 0.9
    }),
    loading: false,
    error: null,
    response: null
  })
}));

describe('ChatbotWidget', () => {
  test('renders chatbot widget', () => {
    render(<ChatbotWidget />);

    // Initially shows float button
    expect(screen.getByText('Robotics Assistant')).toBeInTheDocument();
  });

  test('toggles chat window', () => {
    render(<ChatbotWidget />);

    const floatButton = screen.getByText('Robotics Assistant');
    fireEvent.click(floatButton);

    expect(screen.getByText('Robotics Assistant')).toBeInTheDocument(); // Header text
  });

  test('sends message', async () => {
    render(<ChatbotWidget />);

    // Open chat
    fireEvent.click(screen.getByText('Robotics Assistant'));

    // Type and send message
    const input = screen.getByPlaceholderText('Ask about robotics concepts...');
    fireEvent.change(input, { target: { value: 'Hello' } });

    const sendButton = screen.getByText('Send');
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(screen.getByText('Hello')).toBeInTheDocument();
    });
  });
});
```

## Summary

This UI integration provides a seamless chatbot experience within the Docusaurus documentation interface. Students can ask questions about robotics concepts and receive contextually relevant answers based on the textbook content, with proper attribution to sources and confidence indicators.

Continue with [Deployment](./deployment) to learn about deploying the complete RAG chatbot system.