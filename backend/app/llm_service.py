"""
LLM Service for RAG Chatbot
Handles language model operations for chat generation.
This service integrates with Google's Gemini API.
"""
from typing import List, Dict, Optional
import logging
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service class for handling language model operations.
    Uses Google's Gemini API for generating responses based on context.
    """

    def __init__(self):
        """
        Initialize the LLM service.
        Initializes connection to Google's Gemini API.
        """
        logger.info("Initializing LLM Service with Google Gemini")

        # Get Gemini API key from environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        # Configure the Gemini API
        genai.configure(api_key=gemini_api_key)

        # Initialize the model
        self.model = genai.GenerativeModel('gemini-2.0-flash')

        logger.info("Gemini model initialized successfully")

    def generate_response(self, query: str, context: Optional[List[Dict]] = None, history: Optional[List[Dict]] = None) -> str:
        """
        Generate a response based on the query, context, and conversation history.

        Args:
            query: The user's query
            context: Retrieved documents or context to inform the response
            history: Conversation history for context

        Returns:
            Generated response string
        """
        logger.info(f"Generating response for query: {query[:50]}...")

        # Build a prompt that includes the query, context, and history
        prompt = self._build_prompt(query, context, history)

        try:
            # Generate response using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                }
            )

            result = response.text
            logger.info("Response generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {str(e)}")

            # If there's context available, return a response based on the context even if API fails
            if context and len(context) > 0:
                # Extract relevant information from the context
                context_snippets = []
                for item in context:
                    content = item.get('content', '')[:300]  # Take first 300 chars
                    source = item.get('metadata', {}).get('title', 'Unknown source')
                    if content.strip():
                        context_snippets.append(f"From {source}: {content}...")

                if context_snippets:
                    context_preview = "\n\n".join(context_snippets[:2])  # Show first 2 snippets
                    return f"I found relevant information in the robotics textbook:\n\n{context_preview}\n\nHowever, I encountered an issue generating a detailed response due to API quota limits. The information above is from the textbook content you've ingested."

            return f"Sorry, I encountered an error processing your request: {str(e)}"

    def _build_prompt(self, query: str, context: Optional[List[Dict]], history: Optional[List[Dict]]) -> str:
        """
        Build a prompt for the LLM that includes context and history.

        Args:
            query: The user's query
            context: Retrieved context documents
            history: Conversation history

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        # Add system context
        prompt_parts.append("You are a helpful robotics education assistant. Answer questions based on the provided context from the textbook. Be specific and cite which parts of the textbook you're referencing when possible.")

        # Add context if available
        if context:
            prompt_parts.append("\nHere are relevant excerpts from the robotics textbook:")
            for i, ctx in enumerate(context):
                content = ctx.get('content', '')[:1000]  # Limit context length
                source = ctx.get('metadata', {}).get('title', 'Unknown source')
                prompt_parts.append(f"\n{i+1}. From {source}:\n{content}")

        # Add conversation history if available
        if history:
            prompt_parts.append("\nPrevious conversation:")
            for turn in history:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                prompt_parts.append(f"{role.capitalize()}: {content}")

        # Add the current query
        prompt_parts.append(f"\nCurrent question: {query}")
        prompt_parts.append("\nPlease answer the question based on the provided context and previous conversation.")

        return "\n".join(prompt_parts)


# Global instance of the LLM service
# In a real application, this might be managed by a dependency injection framework
llm_service = None

def get_llm_service():
    """
    Get the global LLM service instance.
    Initializes the service if it doesn't exist.
    """
    global llm_service
    if llm_service is None:
        try:
            llm_service = LLMService()
        except ValueError as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise
    return llm_service