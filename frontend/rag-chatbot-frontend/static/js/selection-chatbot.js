// selection-chatbot.js - Text selection to chatbot functionality

(function() {
    'use strict';

    // Configuration
    const CHATBOT_BUTTON_ID = 'selection-chatbot-button';
    const CHATBOT_IFRAME_ID = 'chatbot-iframe';
    const SELECTION_THRESHOLD = 1; // Minimum characters to show button (now any selected text)
    const BUTTON_TIMEOUT = 3000; // Hide button after 3 seconds of inactivity

    // State variables
    let selectionButton = null;
    let chatbotIframe = null;
    let hideTimeout = null;
    let isChatbotVisible = false;
    let lastSelectedText = null; // Track the last selected text to keep button visible
    let lastButtonPosition = null; // Track the last position where button was shown

    // Create the selection chat button
    function createSelectionButton() {
        if (document.getElementById(CHATBOT_BUTTON_ID)) {
            return; // Already exists
        }

        selectionButton = document.createElement('div');
        selectionButton.id = CHATBOT_BUTTON_ID;
        selectionButton.innerHTML = 'Ask Chatbot';
        selectionButton.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: auto;
            height: auto;
            min-width: 30px;
            min-height: 30px;
            padding: 8px 12px;
            background: #4f46e5;
            color: white;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            z-index: 10000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            user-select: none;
            pointer-events: auto;
            transition: transform 0.2s, opacity 0.2s;
            opacity: 0;
            transform: scale(0);
            font-family: system-ui, -apple-system, sans-serif;
            white-space: nowrap;
        `;

        // Add hover effect
        selectionButton.addEventListener('mouseenter', () => {
            selectionButton.style.background = '#4338ca';
            selectionButton.style.transform = 'scale(1.1)';
        });

        selectionButton.addEventListener('mouseleave', () => {
            selectionButton.style.background = '#4f46e5';
            selectionButton.style.transform = 'scale(1)';
        });

        // Click handler
        selectionButton.addEventListener('click', handleButtonClick);

        // Add to document
        document.body.appendChild(selectionButton);
    }

    // Handle button click
    function handleButtonClick() {
        const selectedText = getSelectedText();
        if (selectedText.trim()) {
            // Clear any hide timeout when button is clicked
            if (hideTimeout) {
                clearTimeout(hideTimeout);
                hideTimeout = null;
            }
            openChatbot(selectedText.trim());
        }
    }

    // Get currently selected text
    function getSelectedText() {
        const selection = window.getSelection();
        return selection.toString().trim();
    }

    // Open chatbot with selected text
    function openChatbot(text) {
        // Clear any hide timeout when opening chatbot
        if (hideTimeout) {
            clearTimeout(hideTimeout);
            hideTimeout = null;
        }

        // First, try to find and click the floating chat button to ensure it's open
        let floatingChatButton = document.querySelector('.floating-chat-button') ||
                                 document.querySelector('[aria-label="Open chat"]');

        if (floatingChatButton) {
            // Click the button to open the chat if it's closed
            floatingChatButton.click();
        }

        // Send the selected text via postMessage to the window
        // The React FloatingChat component is listening for these messages
        window.postMessage({
            type: 'SELECTED_TEXT',
            text: text
        }, '*');

        isChatbotVisible = true;

        // Hide the selection button after opening chatbot
        hideSelectionButton();
    }

    // Create chatbot iframe as a last resort (fallback if React component doesn't exist)
    function createChatbotIframe(selectedText = '') {
        // Try to find the floating chat button first and use it instead of creating an iframe
        let floatingChatButton = document.querySelector('.floating-chat-button') ||
                                 document.querySelector('[aria-label="Open chat"]');

        if (floatingChatButton) {
            // Click the button to open the chat if it's closed
            floatingChatButton.click();

            // Send the selected text via postMessage to the window
            window.postMessage({
                type: 'SELECTED_TEXT',
                text: selectedText
            }, '*');

            isChatbotVisible = true;
            return; // Exit early since we're using the existing React component
        }

        // If no floating chat button exists, create the iframe as a fallback
        // Remove existing iframe if any
        const existingIframe = document.getElementById(CHATBOT_IFRAME_ID);
        if (existingIframe) {
            existingIframe.remove();
        }

        // Create iframe container
        const iframeContainer = document.createElement('div');
        iframeContainer.id = 'chatbot-container';
        iframeContainer.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 400px;
            height: 500px;
            z-index: 10001;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            overflow: hidden;
        `;

        // Create iframe
        chatbotIframe = document.createElement('iframe');
        chatbotIframe.id = CHATBOT_IFRAME_ID;
        chatbotIframe.className = 'chatbot-iframe';

        // Get the backend URL from environment or default
        const backendUrl = window.REACT_APP_BACKEND_URL ||
                          window.env?.REACT_APP_BACKEND_URL ||
                          'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app/';

        // Load the main page, not the API endpoint
        chatbotIframe.src = backendUrl;
        chatbotIframe.style.cssText = `
            width: 100%;
            height: 100%;
            border: none;
        `;

        iframeContainer.appendChild(chatbotIframe);
        document.body.appendChild(iframeContainer);

        // Wait for iframe to load, then send the selected text
        chatbotIframe.onload = function() {
            // Small delay to ensure the page is fully loaded before sending the message
            setTimeout(() => {
                // Send to the iframe's content window if needed as a last resort
                chatbotIframe.contentWindow.postMessage({
                    type: 'SELECTED_TEXT',
                    text: selectedText
                }, '*');
            }, 500);
        };

        isChatbotVisible = true;
    }

    // Show the selection button near the selected text
    function showSelectionButton() {
        if (!selectionButton) {
            createSelectionButton();
        }

        const selection = window.getSelection();
        const hasValidSelection = selection.toString().trim() && selection.toString().length >= SELECTION_THRESHOLD;

        // If we have a valid selection, position and show the button
        if (hasValidSelection && selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();

            // Store the position for later use when selection is cleared
            lastButtonPosition = {
                top: (rect.top + window.scrollY - 60) + 'px',
                left: (rect.left + window.scrollX + rect.width/2 - 25) + 'px'
            };

            // Position button near the selection
            selectionButton.style.top = lastButtonPosition.top;
            selectionButton.style.left = lastButtonPosition.left;

            // Show button with animation
            selectionButton.style.opacity = '1';
            selectionButton.style.transform = 'scale(1)';

            // Clear any existing timeout
            if (hideTimeout) {
                clearTimeout(hideTimeout);
            }
        } else if (!hasValidSelection && lastButtonPosition) {
            // If there's no selection but we have a last position, keep the button visible at that position
            selectionButton.style.top = lastButtonPosition.top;
            selectionButton.style.left = lastButtonPosition.left;

            // Ensure the button remains visible
            selectionButton.style.opacity = '1';
            selectionButton.style.transform = 'scale(1)';
        }
    }

    // Hide the selection button
    function hideSelectionButton() {
        if (selectionButton) {
            selectionButton.style.opacity = '0';
            selectionButton.style.transform = 'scale(0)';
        }
    }

    // Handle text selection
    function handleTextSelection() {
        const selectedText = getSelectedText();

        if (selectedText && selectedText.length >= SELECTION_THRESHOLD) {
            lastSelectedText = selectedText; // Store the selected text
            showSelectionButton();
        } else {
            // Clear any existing timeout but don't hide immediately
            // This allows the button to stay visible after selection
            if (hideTimeout) {
                clearTimeout(hideTimeout);
            }
            // Set a timeout to hide the button after BUTTON_TIMEOUT milliseconds
            // if no interaction happens with the button
            hideTimeout = setTimeout(() => {
                hideSelectionButton();
                lastSelectedText = null; // Clear the stored text when hiding
                lastButtonPosition = null; // Clear the stored position when hiding
            }, BUTTON_TIMEOUT);
        }
    }

    // Mobile long-press detection
    let longPressTimer = null;
    let startX, startY;

    function handleTouchStart(e) {
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;

        longPressTimer = setTimeout(() => {
            // Simulate text selection at touch position
            const element = document.elementFromPoint(startX, startY);
            if (element && element.textContent && element.textContent.length > SELECTION_THRESHOLD) {
                // Create a temporary selection
                const selection = window.getSelection();
                const range = document.createRange();
                range.selectNodeContents(element);
                selection.removeAllRanges();
                selection.addRange(range);

                showSelectionButton();
            }
        }, 600); // 600ms for long press
    }

    function handleTouchMove(e) {
        if (Math.abs(e.touches[0].clientX - startX) > 10 ||
            Math.abs(e.touches[0].clientY - startY) > 10) {
            clearTimeout(longPressTimer);
        }
    }

    function handleTouchEnd() {
        clearTimeout(longPressTimer);
    }

    // Initialize the functionality
    function init() {
        // Create the button
        createSelectionButton();

        // Add event listeners for desktop
        document.addEventListener('mouseup', function() {
            // Add a small delay to ensure the selection is complete when mouse is released
            setTimeout(handleTextSelection, 10);
        });
        document.addEventListener('selectionchange', handleTextSelection);

        // Add event listeners for mobile
        document.addEventListener('touchstart', handleTouchStart, { passive: false });
        document.addEventListener('touchmove', handleTouchMove, { passive: false });
        document.addEventListener('touchend', handleTouchEnd);

        // Add event listener to hide button when clicking elsewhere
        document.addEventListener('click', function(e) {
            if (e.target !== selectionButton &&
                !selectionButton.contains(e.target) &&
                e.target.tagName !== 'IFRAME') {
                hideSelectionButton();
                // Clear any hide timeout when clicking elsewhere
                if (hideTimeout) {
                    clearTimeout(hideTimeout);
                    hideTimeout = null;
                }
            }
        });

        // Add keyboard shortcut (Ctrl/Cmd+Shift+C)
        document.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
                e.preventDefault();
                const selectedText = getSelectedText();
                if (selectedText.trim()) {
                    openChatbot(selectedText.trim());
                } else {
                    alert('Please select some text first, then press Ctrl/Cmd+Shift+C');
                }
            }
        });
    }

    // Initialize when DOM is loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Export for debugging purposes
    window.SelectionChatbot = {
        showButton: showSelectionButton,
        hideButton: hideSelectionButton,
        openChatbot: openChatbot,
        getSelectedText: getSelectedText
    };
})();