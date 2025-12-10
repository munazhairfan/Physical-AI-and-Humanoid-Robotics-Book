"""
LLM Service for RAG Chatbot
Handles language model operations for chat generation.
This service integrates with Google's Gemini API.
"""
from typing import List, Dict, Optional
import logging
import os
from dotenv import load_dotenv

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
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY environment variable not set - will use mock functionality")
            self.model = None
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                # Initialize the model
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Gemini model initialized successfully with API key")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {str(e)} - will use mock functionality")
                self.model = None

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

        if self.model:
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
                logger.info("Response generated successfully with Gemini")
                return result
            except Exception as e:
                logger.error(f"Error generating response with Gemini: {str(e)}")
        else:
            logger.warning("LLM service not initialized with API key, using fallback response")

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
                # Generate a more meaningful response based on context
                return f"Based on the robotics textbook:\n\n{context_preview}\n\nHow can I help you understand this better?"

        # Generate a more helpful response based on the query content
        lower_query = query.lower()
        if 'hello' in lower_query or 'hi' in lower_query or 'hey' in lower_query:
            return "Hello! I'm your Robotics Assistant. I'm currently running in demo mode without the full AI backend configured. How can I help you with robotics concepts today?"
        elif 'robot' in lower_query or 'ai' in lower_query:
            return "Robots and AI are fascinating fields! Physical AI involves creating intelligent systems that interact with the physical world. Humanoid robots specifically mimic human form and behavior. What specific aspect would you like to know more about?"
        elif 'learn' in lower_query or 'study' in lower_query or 'education' in lower_query:
            return "Learning about robotics involves understanding mechanics, electronics, programming, and artificial intelligence. Key areas include kinematics, control systems, sensors, actuators, and machine learning. Would you like to dive deeper into any specific area?"
        elif 'humanoid' in lower_query or 'bipedal' in lower_query or 'walking' in lower_query:
            return "Humanoid robots are designed to resemble and mimic human behavior. Key challenges include balance control, gait planning, and natural movement. They often use inverse kinematics and PID controllers for smooth motion. What would you like to know about humanoid robotics?"
        elif 'control' in lower_query or 'motion' in lower_query or 'movement' in lower_query:
            return "Robot control involves various techniques like PID controllers, inverse kinematics, and trajectory planning. For humanoid robots, maintaining balance and creating natural movements require sophisticated control algorithms. Would you like to know about a specific control method?"
        else:
            responses = [
                "That's an interesting question about robotics! Physical AI and humanoid robotics involve complex interactions between mechanical systems, sensors, and artificial intelligence. Could you elaborate on what specifically interests you?",
                "Robots are amazing systems that combine mechanics, electronics, and software. In humanoid robotics, we focus on creating machines that can interact with humans and environments in human-like ways. What aspect would you like to explore?",
                "Great question! Robotics encompasses many fields including kinematics, dynamics, control systems, and AI. Physical AI specifically deals with robots that interact with the physical world. What specific area interests you most?",
                "I appreciate your interest in robotics technology! There are many fascinating aspects to explore, from the mechanics of robot movement to the AI that powers their decision-making. What would you like to know more about?",
                "Robotics is an interdisciplinary field combining engineering and computer science. Humanoid robots add the complexity of mimicking human form and function. What specific topic would you like to discuss?"
            ]
            import random
            return responses[random.randint(0, len(responses) - 1)]

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
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            # Create a service instance even if initialization fails
            llm_service = LLMService()  # This will handle the error internally
    return llm_service