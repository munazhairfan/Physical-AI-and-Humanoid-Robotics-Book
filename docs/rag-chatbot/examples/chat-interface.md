---
title: Chat Interface Example
sidebar_label: Chat Interface
---

# Chat Interface Example

This example demonstrates how to implement and use the chat interface with the RAG Chatbot, including conversation history, context retrieval, and response generation.

## Overview

The chat interface allows users to have conversations with the RAG Chatbot, which:
1. Takes user messages and conversation history
2. Retrieves relevant context from the vector store
3. Generates contextual responses using the LLM
4. Maintains conversation flow

## Prerequisites

- Backend API running (typically at `http://localhost:8000`)
- Access to the `/chat` endpoint
- Documents already embedded in the vector store

## Step 1: Basic Chat Request

Here's how to send a simple chat message to the backend:

### Python Example

```python
import requests
import json

def send_chat_message(api_url: str, message: str, history: list = None):
    """
    Send a chat message to the RAG Chatbot backend API.

    Args:
        api_url: Base URL of the backend API
        message: The user's message
        history: Conversation history (list of previous messages)

    Returns:
        Response from the chat API
    """
    # Prepare the request payload
    payload = {
        "message": message,
        "history": history or []
    }

    # Make the API request
    response = requests.post(
        f"{api_url}/chat",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Bot response: {result['response']}")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Example usage
api_url = "http://localhost:8000"  # Replace with your backend URL
user_message = "What is artificial intelligence?"
chat_response = send_chat_message(api_url, user_message)

if chat_response:
    print(f"Bot: {chat_response['response']}")
```

### JavaScript/TypeScript Example

```javascript
async function sendChatMessage(apiUrl, message, history = []) {
    try {
        const response = await fetch(`${apiUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: history
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log(`Bot response: ${result.response}`);
        return result;
    } catch (error) {
        console.error('Error sending chat message:', error);
        throw error;
    }
}

// Example usage
const apiUrl = "http://localhost:8000"; // Replace with your backend URL
const userMessage = "What is artificial intelligence?";
sendChatMessage(apiUrl, userMessage)
    .then(response => {
        console.log(`Bot: ${response.response}`);
    });
```

## Step 2: Maintaining Conversation History

To maintain context across multiple messages, you need to keep track of the conversation history:

```python
class ChatSession:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.history = []

    def send_message(self, message: str):
        """
        Send a message and update the conversation history.

        Args:
            message: The user's message

        Returns:
            The bot's response
        """
        # Prepare the request with current history
        payload = {
            "message": message,
            "history": self.history
        }

        # Make the API request
        response = requests.post(
            f"{self.api_url}/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )

        if response.status_code == 200:
            result = response.json()

            # Update history with the new exchange
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": result['response']})

            print(f"User: {message}")
            print(f"Bot: {result['response']}")
            print()

            return result['response']
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def get_history(self):
        """Get the current conversation history."""
        return self.history

    def clear_history(self):
        """Clear the conversation history."""
        self.history = []

# Example usage
chat_session = ChatSession("http://localhost:8000")  # Replace with your backend URL

# First message
chat_session.send_message("What is machine learning?")

# Follow-up message (with context from previous conversation)
chat_session.send_message("Can you give me some examples?")

# Check the history
print("Conversation history:")
for i, message in enumerate(chat_session.get_history()):
    role = message['role']
    content = message['content']
    print(f"{i+1}. {role}: {content[:50]}...")
```

## Step 3: Complete Chat Interface Example

Here's a complete example that implements a full chat interface:

```python
import requests
import json
import os
from typing import List, Dict

class RAGChatInterface:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.history = []
        self.session_id = None

    def send_message(self, message: str) -> str:
        """
        Send a message to the chatbot and return the response.

        Args:
            message: The user's message

        Returns:
            The bot's response
        """
        try:
            payload = {
                "message": message,
                "history": self.history
            }

            response = requests.post(
                f"{self.api_url}/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            # Update history
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": result['response']})

            return result['response']

        except requests.exceptions.RequestException as e:
            error_msg = f"Error communicating with chatbot: {str(e)}"
            print(error_msg)
            return error_msg
        except KeyError:
            error_msg = "Invalid response format from chatbot"
            print(error_msg)
            return error_msg

    def start_conversation(self):
        """Start a new conversation with a welcome message."""
        welcome_msg = "Hello! I'm your RAG Chatbot. I can answer questions based on the documents I've been trained on. How can I help you today?"
        self.history.append({"role": "assistant", "content": welcome_msg})
        return welcome_msg

    def reset_conversation(self):
        """Reset the conversation history."""
        self.history = []
        return self.start_conversation()

    def display_conversation(self, limit: int = 10):
        """Display the last N messages in the conversation."""
        messages_to_show = self.history[-limit:] if len(self.history) > limit else self.history

        print(f"\n--- Last {len(messages_to_show)} messages ---")
        for i, msg in enumerate(messages_to_show):
            role = "👤 You" if msg['role'] == 'user' else "🤖 Bot"
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"{role}: {content}")
        print("---\n")

def main():
    """Main function to run the chat interface."""
    print("Starting RAG Chatbot Interface...")
    print("Type 'quit' to exit, 'history' to see recent messages, 'reset' to start over\n")

    # Initialize the chat interface
    chat = RAGChatInterface("http://localhost:8000")  # Replace with your backend URL

    # Start with a welcome message
    welcome = chat.start_conversation()
    print(f"🤖 Bot: {welcome}\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'reset':
                welcome = chat.reset_conversation()
                print(f"🤖 Bot: {welcome}")
            elif user_input.lower() == 'history':
                chat.display_conversation()
            elif user_input:
                response = chat.send_message(user_input)
                print(f"🤖 Bot: {response}")
            else:
                print("Please enter a message.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
```

## Step 4: Advanced Chat Features

### Context-Aware Responses

The RAG Chatbot can provide context-aware responses by retrieving relevant information:

```python
def advanced_chat_query(api_url: str, message: str, history: list = None, context_filter: dict = None):
    """
    Send a chat message with additional context filtering.

    Args:
        api_url: Base URL of the backend API
        message: The user's message
        history: Conversation history
        context_filter: Optional filters for context retrieval

    Returns:
        Response from the chat API
    """
    payload = {
        "message": message,
        "history": history or []
    }

    # In a real implementation, you might have additional parameters
    # for specifying context filters or sources
    response = requests.post(
        f"{api_url}/chat",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
```

### Handling Different Message Types

```python
def handle_special_messages(chat_interface: RAGChatInterface, message: str):
    """
    Handle special commands or message types.

    Args:
        chat_interface: The chat interface instance
        message: The user's message

    Returns:
        True if the message was handled as a special command, False otherwise
    """
    message_lower = message.lower().strip()

    if message_lower.startswith('/'):
        # Handle slash commands
        if message_lower == '/help':
            help_text = """
Available commands:
/help - Show this help message
/history - Show recent conversation
/reset - Start a new conversation
/stats - Show conversation statistics
            """
            print(help_text)
            return True
        elif message_lower == '/stats':
            print(f"Conversation length: {len(chat_interface.history)} messages")
            print(f"Session ID: {chat_interface.session_id or 'Not set'}")
            return True
        elif message_lower == '/history':
            chat_interface.display_conversation()
            return True
        elif message_lower == '/reset':
            welcome = chat_interface.reset_conversation()
            print(f"🤖 Bot: {welcome}")
            return True

    return False

# Example usage in the main loop
def enhanced_main():
    chat = RAGChatInterface("http://localhost:8000")
    welcome = chat.start_conversation()
    print(f"🤖 Bot: {welcome}\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Check if it's a special command
            if handle_special_messages(chat, user_input):
                continue

            # Regular chat message
            if user_input:
                response = chat.send_message(user_input)
                print(f"🤖 Bot: {response}")
            else:
                print("Please enter a message.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
```

## Step 5: Frontend Integration Example

Here's how to integrate the chat interface into a web frontend:

```html
<!DOCTYPE html>
<html>
<head>
    <title>RAG Chatbot Interface</title>
    <style>
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-family: Arial, sans-serif;
        }

        .chat-messages {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f9f9f9;
        }

        .message {
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 5px;
        }

        .user-message {
            background-color: #d1ecf1;
            text-align: right;
        }

        .bot-message {
            background-color: #f8d7da;
        }

        .input-container {
            display: flex;
        }

        .input-container input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ccc;
        }

        .input-container button {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>RAG Chatbot</h1>
        <div id="chat-messages" class="chat-messages"></div>
        <div class="input-container">
            <input type="text" id="user-input" placeholder="Type your message here..." />
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-input');
        let conversationHistory = [];

        // Add a message to the chat display
        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message');
            messageDiv.classList.add(role + '-message');

            const messageText = document.createElement('div');
            messageText.textContent = content;

            messageDiv.appendChild(messageText);
            chatMessages.appendChild(messageDiv);

            // Scroll to the bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Send a message to the backend
        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            // Add user message to UI
            addMessage('user', message);
            userInput.value = '';

            // Disable input while waiting for response
            userInput.disabled = true;

            try {
                const response = await fetch('http://localhost:8000/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        history: conversationHistory
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();

                // Add bot response to UI
                addMessage('bot', data.response);

                // Update conversation history
                conversationHistory.push({role: 'user', content: message});
                conversationHistory.push({role: 'assistant', content: data.response});

            } catch (error) {
                console.error('Error sending message:', error);
                addMessage('bot', 'Sorry, I encountered an error. Please try again.');
            } finally {
                userInput.disabled = false;
                userInput.focus();
            }
        }

        // Allow sending with Enter key
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Initial welcome message
        addMessage('bot', 'Hello! I\'m your RAG Chatbot. How can I help you today?');
    </script>
</body>
</html>
```

## API Response Format

The `/chat` endpoint returns a JSON response in this format:

```json
{
  "response": "The generated response from the chatbot based on the user's message and conversation history."
}
```

## Error Handling

Always implement proper error handling for chat interactions:

```python
def chat_with_error_handling(api_url: str, message: str, history: list = None):
    """
    Send a chat message with comprehensive error handling.
    """
    try:
        payload = {
            "message": message,
            "history": history or []
        }

        response = requests.post(
            f"{api_url}/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30  # 30 second timeout
        )

        # Check if the request was successful
        response.raise_for_status()

        result = response.json()

        # Validate the response structure
        if 'response' not in result:
            raise ValueError("Invalid response format from chat API")

        return result

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        print(f"Response content: {response.text}")
        return {"response": "Sorry, I'm having trouble responding right now. Please try again."}
    except requests.exceptions.ConnectionError:
        print("Connection error: Could not reach the chat API")
        return {"response": "Sorry, I'm currently unavailable. Please check your connection."}
    except requests.exceptions.Timeout:
        print("Request timed out: The chat API took too long to respond")
        return {"response": "Sorry, I'm taking too long to respond. Please try again."}
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return {"response": "Sorry, an error occurred. Please try again."}
    except ValueError as e:
        print(f"Value error: {e}")
        return {"response": "Sorry, I received an invalid response. Please try again."}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"response": "Sorry, an unexpected error occurred. Please try again."}
```

This example demonstrates how to implement a complete chat interface with the RAG Chatbot, including conversation history management, error handling, and both backend and frontend integration approaches.