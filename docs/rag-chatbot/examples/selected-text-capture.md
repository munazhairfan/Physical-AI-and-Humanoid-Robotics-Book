---
title: Selected Text Capture Example
sidebar_label: Selected Text Capture
---

# Selected Text Capture Example

This example demonstrates how to capture user-selected text and send it to the RAG Chatbot backend for processing, allowing the chatbot to respond based only on the selected text.

## Overview

The selected text capture feature allows users to:
1. Highlight text on a webpage or document
2. Capture the selected text programmatically
3. Send the selected text to the backend for RAG processing
4. Receive a response focused specifically on the selected text

## Prerequisites

- Backend API running (typically at `http://localhost:8000`)
- Access to the `/selected_text` endpoint
- Frontend environment with JavaScript support

## Step 1: Capture Selected Text in the Browser

Here's how to capture selected text using JavaScript:

### Basic Text Selection Capture

```javascript
function getSelectedText() {
    // Get the currently selected text
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    if (selectedText) {
        console.log('Selected text:', selectedText);
        return selectedText;
    } else {
        console.log('No text selected');
        return null;
    }
}

// Example usage
document.addEventListener('mouseup', function() {
    const selectedText = getSelectedText();
    if (selectedText) {
        console.log('User selected:', selectedText);
        // You can now send this text to your backend
    }
});
```

### Enhanced Text Selection with Additional Information

```javascript
function getSelectedTextWithInfo() {
    const selection = window.getSelection();

    if (selection.toString().trim()) {
        const range = selection.getRangeAt(0);
        const selectedText = selection.toString().trim();

        // Get additional context
        const startContainer = range.startContainer;
        const endContainer = range.endContainer;

        return {
            text: selectedText,
            startOffset: range.startOffset,
            endOffset: range.endOffset,
            startContainer: startContainer.textContent,
            endContainer: endContainer.textContent,
            element: startContainer.parentElement || startContainer
        };
    }

    return null;
}

// Example usage
document.addEventListener('mouseup', function() {
    const selectionInfo = getSelectedTextWithInfo();
    if (selectionInfo) {
        console.log('Selection details:', selectionInfo);
    }
});
```

## Step 2: Send Selected Text to Backend

Here's how to send the captured selected text to the backend:

### JavaScript Example

```javascript
async function sendSelectedTextToBackend(selectedText, apiUrl = 'http://localhost:8000') {
    try {
        const response = await fetch(`${apiUrl}/selected_text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: selectedText
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Backend response:', result);
        return result;
    } catch (error) {
        console.error('Error sending selected text to backend:', error);
        throw error;
    }
}

// Complete example combining selection and sending
function setupSelectedTextHandler(apiUrl = 'http://localhost:8000') {
    document.addEventListener('mouseup', async function() {
        const selectedText = getSelectedText();

        if (selectedText && selectedText.length > 0) {
            console.log('Sending selected text to backend:', selectedText.substring(0, 50) + '...');

            try {
                const response = await sendSelectedTextToBackend(selectedText, apiUrl);
                console.log('Received response:', response.response);

                // You can now display the response to the user
                displaySelectedTextResponse(response.response, selectedText);
            } catch (error) {
                console.error('Failed to process selected text:', error);
                displaySelectedTextResponse('Sorry, there was an error processing your selected text.', selectedText);
            }
        }
    });
}

// Example response display function
function displaySelectedTextResponse(response, originalSelection) {
    // Create or update a UI element to show the response
    let responseDiv = document.getElementById('selected-text-response');

    if (!responseDiv) {
        responseDiv = document.createElement('div');
        responseDiv.id = 'selected-text-response';
        responseDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            max-width: 400px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 0.375rem;
            padding: 1rem;
            z-index: 10000;
            font-family: Arial, sans-serif;
            font-size: 14px;
            box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
        `;
        document.body.appendChild(responseDiv);
    }

    responseDiv.innerHTML = `
        <strong>Selected Text Response:</strong><br>
        <em>"${originalSelection.substring(0, 100)}${originalSelection.length > 100 ? '...' : ''}"</em><br><br>
        ${response}
        <button onclick="this.parentElement.style.display='none'"
                style="margin-top: 10px; padding: 5px 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
            Close
        </button>
    `;
}
```

### Python Backend Integration Example

If you're testing the backend directly:

```python
import requests
import json

def test_selected_text_endpoint(api_url: str, selected_text: str):
    """
    Test the selected_text endpoint directly.

    Args:
        api_url: Base URL of the backend API
        selected_text: The text that was selected by the user

    Returns:
        Response from the selected_text API
    """
    payload = {
        "text": selected_text
    }

    response = requests.post(
        f"{api_url}/selected_text",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Selected text response: {result['response']}")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Example usage
api_url = "http://localhost:8000"  # Replace with your backend URL
selected_text = "Artificial intelligence is a wonderful field of study."
response = test_selected_text_endpoint(api_url, selected_text)
```

## Step 3: Complete Selected Text Capture Widget

Here's a complete implementation of a selected text capture widget:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Selected Text Capture Example</title>
    <style>
        .sample-content {
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            line-height: 1.6;
            font-family: Arial, sans-serif;
        }

        .selection-indicator {
            position: fixed;
            top: 10px;
            right: 10px;
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 4px;
            padding: 10px;
            max-width: 300px;
            display: none;
            z-index: 10000;
        }

        .selection-indicator button {
            margin-top: 5px;
            padding: 5px 10px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }

        .response-panel {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 20px;
            max-width: 500px;
            display: none;
            z-index: 10001;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .response-panel .close-btn {
            position: absolute;
            top: 10px;
            right: 15px;
            background: none;
            border: none;
            font-size: 1.5em;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="sample-content">
        <h1>Sample Document for Text Selection</h1>
        <p>
            Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
        </p>
        <p>
            Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
        </p>
        <p>
            Modern machine learning techniques are at the heart of AI. Problems for AI applications include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects.
        </p>
    </div>

    <div id="selection-indicator" class="selection-indicator">
        <strong>Selected Text:</strong><br>
        <span id="selected-text-preview"></span><br>
        <button id="process-selection">Ask AI about this text</button>
    </div>

    <div id="response-panel" class="response-panel">
        <button class="close-btn">&times;</button>
        <h3>AI Response</h3>
        <div id="response-content"></div>
    </div>

    <script>
        let currentSelection = null;

        // Function to get selected text
        function getSelectedText() {
            const selection = window.getSelection();
            return selection.toString().trim();
        }

        // Show selection indicator when text is selected
        document.addEventListener('mouseup', function() {
            const selectedText = getSelectedText();
            const indicator = document.getElementById('selection-indicator');

            if (selectedText) {
                currentSelection = selectedText;
                document.getElementById('selected-text-preview').textContent =
                    selectedText.length > 100 ? selectedText.substring(0, 100) + '...' : selectedText;
                indicator.style.display = 'block';

                // Position the indicator near the cursor
                const rect = document.getSelection().getRangeAt(0).getBoundingClientRect();
                indicator.style.top = (rect.top - 40) + 'px';
                indicator.style.left = (rect.left) + 'px';
            } else {
                indicator.style.display = 'none';
                currentSelection = null;
            }
        });

        // Process the selected text when button is clicked
        document.getElementById('process-selection').addEventListener('click', async function() {
            if (currentSelection) {
                const indicator = document.getElementById('selection-indicator');
                indicator.style.display = 'none';

                try {
                    // Show loading state
                    const responseContent = document.getElementById('response-content');
                    responseContent.innerHTML = 'Processing your selected text...';
                    document.getElementById('response-panel').style.display = 'block';

                    // Send to backend
                    const response = await fetch('http://localhost:8000/selected_text', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: currentSelection
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();

                    // Display the response
                    responseContent.innerHTML = `
                        <strong>Selected Text:</strong><br>
                        "${currentSelection}"<br><br>
                        <strong>AI Response:</strong><br>
                        ${result.response}
                    `;
                } catch (error) {
                    console.error('Error processing selected text:', error);
                    document.getElementById('response-content').innerHTML =
                        'Sorry, there was an error processing your selected text. Please try again.';
                }
            }
        });

        // Close response panel
        document.querySelector('.response-panel .close-btn').addEventListener('click', function() {
            document.getElementById('response-panel').style.display = 'none';
        });

        // Hide indicator if user clicks elsewhere
        document.addEventListener('click', function(e) {
            const indicator = document.getElementById('selection-indicator');
            const isClickInside = indicator.contains(e.target);
            const isSelectionButton = e.target.id === 'process-selection';

            if (!isClickInside && !isSelectionButton && currentSelection) {
                setTimeout(() => {
                    if (!getSelectedText()) {
                        indicator.style.display = 'none';
                        currentSelection = null;
                    }
                }, 10);
            }
        });
    </script>
</body>
</html>
```

## Step 4: Advanced Selected Text Features

### Context-Aware Selection Processing

```javascript
async function processSelectedTextWithContext(selectedText, context = null) {
    try {
        const payload = {
            text: selectedText
        };

        // Include additional context if provided
        if (context) {
            payload.context = context;
        }

        const response = await fetch('http://localhost:8000/selected_text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error processing selected text with context:', error);
        throw error;
    }
}

// Example with document context
function getDocumentContext() {
    return {
        title: document.title,
        url: window.location.href,
        headings: Array.from(document.querySelectorAll('h1, h2, h3')).map(h => h.textContent).slice(0, 5),
        currentScrollPosition: window.scrollY,
        viewportSize: { width: window.innerWidth, height: window.innerHeight }
    };
}

// Enhanced selection handler with context
document.addEventListener('mouseup', async function() {
    const selectedText = getSelectedText();

    if (selectedText && selectedText.length > 0) {
        const context = getDocumentContext();

        try {
            const response = await processSelectedTextWithContext(selectedText, context);
            displaySelectedTextResponse(response.response, selectedText);
        } catch (error) {
            console.error('Failed to process selected text with context:', error);
        }
    }
});
```

### Selection Validation and Processing

```javascript
function validateSelection(selectedText) {
    // Define validation rules
    const minLength = 5;  // Minimum length of selected text
    const maxLength = 2000;  // Maximum length of selected text
    const minWords = 2;  // Minimum number of words

    if (selectedText.length < minLength) {
        return {
            valid: false,
            error: `Selected text is too short (minimum ${minLength} characters)`
        };
    }

    if (selectedText.length > maxLength) {
        return {
            valid: false,
            error: `Selected text is too long (maximum ${maxLength} characters)`
        };
    }

    const wordCount = selectedText.trim().split(/\s+/).length;
    if (wordCount < minWords) {
        return {
            valid: false,
            error: `Selected text has too few words (minimum ${minWords} words)`
        };
    }

    // Check if selection contains mostly non-text content
    const nonTextRatio = (selectedText.match(/[^a-zA-Z0-9\s\.,;:!?'"-]/g) || []).length / selectedText.length;
    if (nonTextRatio > 0.5) {
        return {
            valid: false,
            error: 'Selection contains too much non-text content'
        };
    }

    return { valid: true };
}

// Updated handler with validation
document.addEventListener('mouseup', async function() {
    const selectedText = getSelectedText();

    if (selectedText && selectedText.length > 0) {
        const validation = validateSelection(selectedText);

        if (validation.valid) {
            try {
                const response = await sendSelectedTextToBackend(selectedText);
                displaySelectedTextResponse(response.response, selectedText);
            } catch (error) {
                console.error('Failed to process selected text:', error);
                displaySelectedTextResponse('Sorry, there was an error processing your selected text.', selectedText);
            }
        } else {
            console.log('Selection validation failed:', validation.error);
            // Optionally show validation error to user
            alert(`Cannot process selection: ${validation.error}`);
        }
    }
});
```

## API Response Format

The `/selected_text` endpoint returns a JSON response in this format:

```json
{
  "response": "The generated response from the chatbot focused specifically on the selected text."
}
```

## Backend Implementation Details

The backend endpoint `/selected_text` typically:
1. Receives the selected text
2. Embeds the text using the embedding service
3. Searches for related content in the vector store
4. Generates a response focused on the selected text using the LLM service
5. Returns a contextual response

## Error Handling

Always implement proper error handling for selected text capture:

```javascript
async function safeSelectedTextHandler() {
    try {
        const selectedText = getSelectedText();

        if (!selectedText) {
            console.log('No text selected');
            return;
        }

        // Validate the selection
        const validation = validateSelection(selectedText);
        if (!validation.valid) {
            console.warn('Selection validation failed:', validation.error);
            return;
        }

        // Process with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

        try {
            const response = await fetch('http://localhost:8000/selected_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: selectedText }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            displaySelectedTextResponse(result.response, selectedText);
        } catch (fetchError) {
            clearTimeout(timeoutId);

            if (fetchError.name === 'AbortError') {
                console.error('Request timed out');
                displaySelectedTextResponse('Request timed out. Please try again.', selectedText);
            } else {
                console.error('Fetch error:', fetchError);
                displaySelectedTextResponse('Network error. Please check your connection.', selectedText);
            }
        }
    } catch (error) {
        console.error('Unexpected error in selected text handler:', error);
        displaySelectedTextResponse('An unexpected error occurred. Please try again.', '');
    }
}

// Set up the event listener with error handling
document.addEventListener('mouseup', safeSelectedTextHandler);
```

This example demonstrates how to implement selected text capture functionality that allows users to highlight text and get AI-generated responses focused specifically on that text, with proper error handling and validation.