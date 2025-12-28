"""
Improved LLM Service for RAG Chatbot
Handles language model operations with better context processing for chat generation.
This service integrates with Google's Gemini API through OpenAI-compatible interface.
"""
from typing import List, Dict, Optional
import logging
import os
import re
import html
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def markdown_to_text(markdown_content: str) -> str:
    """
    Convert markdown content to plain text by removing markdown formatting.
    This function is similar to the one in main.py but included here for self-containment.

    Args:
        markdown_content: Raw markdown content

    Returns:
        Plain text content with markdown formatting removed
    """
    if not markdown_content:
        return ""

    # Remove HTML tags if any
    text = html.unescape(markdown_content)

    # Remove common document metadata lines (like 'id:', 'sidebar_position:', etc.)
    text = re.sub(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*.*$', '', text, flags=re.MULTILINE)

    # Remove markdown headers (### Header -> Header)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # Remove bold and italic formatting (**text**, *text*, __text__, _text_)
    text = re.sub(r'\*{2}([^*]+)\*{2}', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)       # *italic*
    text = re.sub(r'_{2}([^_]+)_{2}', r'\1', text)   # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)         # _italic_

    # Remove code blocks (```code```)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # Remove inline code (`code`)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove links [text](url) -> text (including the problematic format)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [text](url) -> text
    text = re.sub(r'\*\*\[([^\]]+)\]\([^)]+\)\*\*', r'\1', text)  # **[text](url)** -> text

    # Remove the specific problematic format in your data
    text = re.sub(r'\*\*\[([^\]]+)\]\([^)]+\)\s*=\s*id:\s*[^\s]+\s+sidebar_position:\s*\d+', '', text)

    # Remove images ![alt](url) -> alt
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)

    # Remove blockquotes
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Remove reference-style links [text][1] and reference definitions [1]: url
    text = re.sub(r'\[([^\]]+)\]\[[^\]]+\]', r'\1', text)  # [text][1] -> text
    text = re.sub(r'\n\[.+\]:.+\n', '\n', text)  # Remove reference definitions

    # Remove YAML frontmatter if present
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)

    # Replace common markdown symbols
    text = re.sub(r'\\', '', text)  # Remove escape characters

    # Remove extra whitespace and normalize
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple blank lines with single
    text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces to single space
    text = text.strip()

    # Clean up any remaining markdown artifacts
    text = re.sub(r'\n\s*-', '\n- ', text)  # Ensure proper list formatting
    text = re.sub(r'\n\s#\s', '\n', text)   # Remove any remaining header markers

    # Clean up any remaining document artifacts
    text = re.sub(r'\.\s*\.\s*\.', '', text)  # Remove ellipsis artifacts
    text = re.sub(r'\s*=\s*id:[^,\n]*,', '', text)  # Remove id assignments
    text = re.sub(r'\s*sidebar_position:\s*\d+', '', text)  # Remove sidebar positions

    return text


class LLMService:
    """
    Improved Service class for handling language model operations.
    Uses Google's Gemini API through OpenAI-compatible interface for generating responses based on context.
    """

    def __init__(self):
        """
        Initialize the LLM service.
        Initializes connection to Google's Gemini API through OpenAI-compatible interface.
        """
        logger.info("Initializing Improved LLM Service with OpenAI-compatible Google Gemini")

        # Get Gemini API key from environment
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY environment variable not set - will use enhanced fallback functionality")
            self.openai_client = None
        else:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(
                    api_key=self.gemini_api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                logger.info("OpenAI client for Gemini initialized successfully with API key")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client for Gemini: {str(e)} - will use enhanced fallback functionality")
                self.openai_client = None

    def generate_response(self, query: str, context: Optional[List[Dict]] = None, history: Optional[List[Dict]] = None) -> str:
        """
        Generate a response based on the query, context, and conversation history.
        Improved to better synthesize information from context.

        Args:
            query: The user's query
            context: Retrieved documents or context to inform the response
            history: Conversation history for context

        Returns:
            Generated response string
        """
        logger.info(f"Generating response for query: {query[:50]}...")

        if self.openai_client:
            # Build a prompt that includes the query, context, and history
            prompt = self._build_improved_prompt(query, context, history)

            try:
                # Generate response using OpenAI-compatible Gemini
                response = self.openai_client.chat.completions.create(
                    model="gemini-2.0-flash",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1000  # Ensure we get complete responses
                )

                result = response.choices[0].message.content
                logger.info("Response generated successfully with OpenAI-compatible Gemini")
                # Clean up the result to ensure proper formatting
                clean_result = ' '.join(result.split())  # Normalize whitespace
                # Remove any remaining markdown artifacts that might have slipped through
                clean_result = markdown_to_text(clean_result)
                return clean_result
            except Exception as e:
                logger.error(f"Error generating response with OpenAI-compatible Gemini: {str(e)}")
                # Even if there's an API error, we should still try to use Qdrant context with enhanced fallback
                logger.info("Falling back to enhanced Qdrant context processing")
        else:
            logger.warning("LLM service not initialized with API key, using enhanced fallback response based on Qdrant context")

        # If there's context available, try to synthesize a better response based on the context
        if context and len(context) > 0:
            # Extract and synthesize relevant information from the context
            synthesized_response = self._synthesize_response_from_context(query, context, history)
            if synthesized_response:
                return synthesized_response

        # If no context is available or synthesis fails, generate a helpful response based on the query content
        return self._generate_fallback_response(query)

    def _synthesize_response_from_context(self, query: str, context: List[Dict], history: Optional[List[Dict]]) -> str:
        """
        Synthesize a response from the retrieved context, focusing on the specific query.
        """
        logger.info(f"Synthesizing response from {len(context)} context items for query: {query[:50]}...")

        # Extract relevant information that matches the query
        query_lower = query.lower()
        relevant_content = []
        all_content = []

        for item in context:
            content = item.get('content', '')
            if content.strip():
                # Convert markdown to plain text and clean up the content
                plain_content = markdown_to_text(content)
                # Remove any remaining artifacts like "=====" or other document formatting
                clean_content = re.sub(r'=+', '', plain_content)  # Remove equals signs
                clean_content = re.sub(r'-+', '', clean_content)  # Remove dash sequences
                clean_content = re.sub(r'\*+', '', clean_content)  # Remove asterisk sequences
                clean_content = re.sub(r'#+', '', clean_content)  # Remove hash sequences
                clean_content = re.sub(r'\s+', ' ', clean_content)  # Normalize whitespace
                clean_content = clean_content.strip()

                # Check if this content is relevant to the query
                content_lower = clean_content.lower()
                all_content.append(clean_content)

                # Simple relevance check - look for query terms in the content
                query_terms = query_lower.split()
                term_matches = sum(1 for term in query_terms if term in content_lower and len(term) > 2)

                if term_matches > 0 or any(keyword in content_lower for keyword in ['robot', 'ai', 'artificial intelligence', 'humanoid', 'control', 'learning', 'sensor', 'motion', 'movement']):
                    relevant_content.append(clean_content)

        # Combine relevant content
        if relevant_content:
            # Join with proper sentence breaks
            combined_content = ". ".join([snippet for snippet in relevant_content if len(snippet) > 20])
            if combined_content and not combined_content.endswith(('.', '!', '?')):
                combined_content += '.'

            # Create a response that directly addresses the query
            if 'ros' in query_lower or 'robot operating system' in query_lower:
                return f"Based on the robotics textbook: {combined_content}\n\nROS (Robot Operating System) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. Would you like to know more about ROS concepts?"
            elif any(keyword in query_lower for keyword in ['control', 'balance', 'walking', 'gait', 'motion', 'movement']):
                return f"Based on the robotics textbook: {combined_content}\n\nHumanoid robot control involves sophisticated techniques for maintaining balance and executing stable walking patterns. The control system must coordinate multiple joints and sensors to achieve stable locomotion. How can I help you understand this better?"
            elif any(keyword in query_lower for keyword in ['challenges', 'future', 'directions', 'development', 'advancement']):
                return f"Based on the robotics textbook: {combined_content}\n\nThe field of humanoid robotics faces several challenges including computational requirements, safety, and robustness. Future directions involve improved autonomy and better human-robot collaboration. How can I help you understand this better?"
            elif any(keyword in query_lower for keyword in ['sensor', 'perception', 'vision', 'camera', 'lidar', 'imu']):
                return f"Based on the robotics textbook: {combined_content}\n\nSensors are crucial for humanoid robots to perceive their environment. They use various sensors like cameras, IMUs, and LiDAR for navigation and interaction. How can I help you understand this better?"
            elif any(keyword in query_lower for keyword in ['ai', 'learning', 'machine learning', 'deep learning', 'neural']):
                return f"Based on the robotics textbook: {combined_content}\n\nAI and machine learning play important roles in enabling humanoid robots to learn and adapt to their environment. How can I help you understand this better?"
            elif any(keyword in query_lower for keyword in ['hardware', 'mechanics', 'actuator', 'motor', 'joint']):
                return f"Based on the robotics textbook: {combined_content}\n\nThe mechanical design of humanoid robots involves sophisticated hardware including actuators, motors, and joints to achieve human-like movement. How can I help you understand this better?"
            else:
                # General response for other queries
                return f"Based on the robotics textbook: {combined_content}\n\nHumanoid robotics is a complex field that combines mechanics, electronics, control systems, and AI. How can I help you understand this better?"

        # If no relevant content found, try to generate a helpful response
        return self._generate_fallback_response(query)

    def _generate_fallback_response(self, query: str) -> str:
        """
        Generate a helpful fallback response when context is not available.
        """
        lower_query = query.lower()
        if 'hello' in lower_query or 'hi' in lower_query or 'hey' in lower_query:
            return "Hello! I'm your Robotics Assistant. I'm here to help you learn about robotics concepts. What would you like to know about robotics?"
        elif any(keyword in lower_query for keyword in ['robot', 'ai', 'artificial intelligence']):
            return "Robots and AI are fascinating fields! Physical AI involves creating intelligent systems that interact with the physical world. Humanoid robots specifically mimic human form and behavior. What specific aspect would you like to know more about?"
        elif any(keyword in lower_query for keyword in ['learn', 'study', 'education', 'book', 'textbook']):
            return "Learning about robotics involves understanding mechanics, electronics, programming, and artificial intelligence. Key areas include kinematics, control systems, sensors, actuators, and machine learning. Would you like to dive deeper into any specific area?"
        elif any(keyword in lower_query for keyword in ['humanoid', 'bipedal', 'walking', 'balance']):
            return "Humanoid robots are designed to resemble and mimic human behavior. Key challenges include balance control, gait planning, and natural movement. They often use inverse kinematics and PID controllers for smooth motion. What would you like to know about humanoid robotics?"
        elif any(keyword in lower_query for keyword in ['control', 'motion', 'movement', 'actuator', 'motor']):
            return "Robot control involves various techniques like PID controllers, inverse kinematics, and trajectory planning. For humanoid robots, maintaining balance and creating natural movements require sophisticated control algorithms. Would you like to know about a specific control method?"
        elif 'ros' in lower_query or 'robot operating system' in lower_query:
            return "ROS (Robot Operating System) is a flexible framework for writing robot software. It provides libraries, tools, and conventions to simplify creating complex robot behavior across various platforms. What specific aspect of ROS would you like to know about?"
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

    def _build_improved_prompt(self, query: str, context: Optional[List[Dict]], history: Optional[List[Dict]]) -> str:
        """
        Build an improved prompt for the LLM that better utilizes context and history.
        """
        prompt_parts = []

        # Add system context
        prompt_parts.append("You are a helpful robotics education assistant. Answer questions based on the provided context from the textbook. Be specific, cite relevant concepts, and provide clear explanations.")

        # Add context if available
        if context:
            prompt_parts.append("\nHere is relevant information from the robotics textbook:")
            # Combine all context snippets into a coherent body of text
            context_texts = []
            for ctx in context:
                content = ctx.get('content', '')
                if content.strip():
                    # Convert markdown to plain text and clean up the content
                    plain_content = markdown_to_text(content)
                    clean_content = ' '.join(plain_content.split())  # Replace multiple spaces/newlines with single spaces
                    # Limit to meaningful content and avoid very short snippets
                    if len(clean_content) > 20:
                        # Try to extract complete sentences if possible
                        snippet = clean_content[:1000]  # Limit context length
                        # Try to find the last sentence ending to avoid cut-off sentences
                        for end_marker in ['.', '!', '?']:
                            last_pos = snippet.rfind(end_marker)
                            if last_pos != -1:  # If we found an end marker
                                snippet = snippet[:last_pos + 1]  # Include the end marker
                                break
                        # If no sentence ending found, try to find word boundaries to avoid cutting words
                        if end_marker not in snippet and len(snippet) == 1000:
                            # Find the last space to avoid cutting in the middle of a word
                            last_space = snippet.rfind(' ')
                            if last_space > len(snippet) * 0.8:  # Only if the space is not too early
                                snippet = snippet[:last_space]

                        # Clean up any remaining artifacts like "rob - siderations" -> "considerations"
                        snippet = snippet.replace(' - ', ' ')
                        snippet = snippet.strip()
                        if snippet:
                            context_texts.append(snippet)

            # Combine all context into a single coherent paragraph
            if context_texts:
                combined_context = ". ".join(context_texts) + "."
                prompt_parts.append(f"\n{combined_context}")

        # Add conversation history if available
        if history:
            prompt_parts.append("\nPrevious conversation:")
            for turn in history:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                prompt_parts.append(f"{role.capitalize()}: {content}")

        # Add the current query with specific instructions
        prompt_parts.append(f"\nCurrent question: {query}")
        prompt_parts.append("\nPlease answer the question based on the provided context and previous conversation. Focus on the specific question asked, synthesize information from the context, and provide a clear, helpful response.")

        return "\n".join(prompt_parts)


# Global instance of the LLM service
# In a real application, this might be managed by a dependency injection framework
llm_service = None

def get_llm_service():
    """
    Get the global improved LLM service instance.
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