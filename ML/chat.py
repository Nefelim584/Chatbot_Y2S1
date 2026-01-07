from __future__ import annotations

from typing import Optional
from loguru import logger
from ML.completion import chat_completion, MistralAPIError
from ML.endpoints import COMPLETION_MODEL_TAGS


class ChatBot:
    """
    A chatbot that manages conversation state and interacts with the Mistral API.
    
    This class maintains message history and provides methods to send messages
    and receive responses from the LLM.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful, friendly, and knowledgeable assistant.",
        model: str = COMPLETION_MODEL_TAGS.get("default", "mistral-small-latest"),
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        safe_prompt: bool = False,
        timeout: float = 30.0,
    ):
        """
        Initialize a new chatbot instance.

        Args:
            system_prompt: The system prompt that defines the assistant's behavior.
            model: The Mistral model to use for completions.
            temperature: Sampling temperature for generation (0.0-2.0).
            max_tokens: Maximum number of tokens to generate (optional).
            safe_prompt: Whether to enable safe prompt filtering.
            timeout: Request timeout in seconds.
        """
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.safe_prompt = safe_prompt
        self.timeout = timeout
        
        # Initialize message history with system prompt
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        
        logger.info(f"ChatBot initialized with model: {model}")

    @property
    def messages(self) -> list[dict[str, str]]:
        """
        Get a copy of the current message history.

        Returns:
            A list of message dictionaries with 'role' and 'content' keys.
        """
        return self._messages.copy()

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation history.

        Args:
            content: The user's message content.
        
        Raises:
            ValueError: If content is empty or None.
        """
        if not content or not content.strip():
            raise ValueError("User message content cannot be empty.")
        
        user_message = {"role": "user", "content": content.strip()}
        self._messages.append(user_message)
        logger.debug(f"Added user message: {content[:50]}...")

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation history.

        This is typically called internally after receiving a response from the API,
        but can also be used to manually add assistant messages if needed.

        Args:
            content: The assistant's message content.
        
        Raises:
            ValueError: If content is empty or None.
        """
        if not content or not content.strip():
            raise ValueError("Assistant message content cannot be empty.")
        
        assistant_message = {"role": "assistant", "content": content.strip()}
        self._messages.append(assistant_message)
        logger.debug(f"Added assistant message: {content[:50]}...")

    def send_message(
        self,
        user_message: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a user message and get a response from the assistant.

        This method adds the user message to the conversation history, calls the
        LLM API, adds the assistant's response to the history, and returns it.

        Args:
            user_message: The user's message to send.
            model: Optional model override (uses instance default if not provided).
            temperature: Optional temperature override.
            max_tokens: Optional max_tokens override.

        Returns:
            The assistant's response as a string.

        Raises:
            ValueError: If user_message is empty.
            MistralAPIError: If the API call fails.
        """
        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty.")
        
        # Add user message to history
        self.add_user_message(user_message)
        
        # Use provided parameters or fall back to instance defaults
        use_model = model if model is not None else self.model
        use_temperature = temperature if temperature is not None else self.temperature
        use_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        logger.info(f"Sending message using model: {use_model}")
        
        try:
            # Call the chat completion API with current message history
            response = chat_completion(
                messages=self._messages,
                model=use_model,
                temperature=use_temperature,
                max_tokens=use_max_tokens,
                safe_prompt=self.safe_prompt,
                timeout=self.timeout,
            )
            
            # Add assistant response to history
            self.add_assistant_message(response)
            
            logger.info(f"Received response (length: {len(response)} chars)")
            return response
            
        except MistralAPIError as exc:
            logger.error(f"Failed to get response from Mistral API: {exc}")
            # Remove the user message from history if the API call failed
            if self._messages and self._messages[-1].get("role") == "user":
                self._messages.pop()
            raise

    def clear_history(self, keep_system: bool = True) -> None:
        """
        Clear the conversation history.

        Args:
            keep_system: If True, keeps the system prompt in the history.
                        If False, clears all messages including the system prompt.
        """
        if keep_system:
            self._messages = [
                {"role": "system", "content": self.system_prompt}
            ]
        else:
            self._messages = []
        
        logger.info("Conversation history cleared")

    def update_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt and replace it in the message history.

        Args:
            new_prompt: The new system prompt.
        
        Raises:
            ValueError: If new_prompt is empty.
        """
        if not new_prompt or not new_prompt.strip():
            raise ValueError("System prompt cannot be empty.")
        
        self.system_prompt = new_prompt.strip()
        
        # Update or add system message in history
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0] = {"role": "system", "content": self.system_prompt}
        else:
            self._messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        logger.info("System prompt updated")

    def get_conversation_summary(self) -> dict:
        """
        Get a summary of the current conversation state.

        Returns:
            A dictionary containing conversation metadata.
        """
        user_messages = sum(1 for msg in self._messages if msg.get("role") == "user")
        assistant_messages = sum(1 for msg in self._messages if msg.get("role") == "assistant")
        
        return {
            "total_messages": len(self._messages),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "model": self.model,
            "system_prompt": self.system_prompt[:100] + "..." if len(self.system_prompt) > 100 else self.system_prompt,
        }


def create_chatbot(
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> ChatBot:
    """
    Factory function to create a ChatBot instance with optional custom settings.

    Args:
        system_prompt: Optional custom system prompt.
        model: Optional model override.
        **kwargs: Additional keyword arguments to pass to ChatBot constructor.

    Returns:
        A new ChatBot instance.
    """
    default_prompt = "You are a helpful, friendly, and knowledgeable assistant."
    default_model = COMPLETION_MODEL_TAGS.get("default", "mistral-small-latest")
    
    return ChatBot(
        system_prompt=system_prompt or default_prompt,
        model=model or default_model,
        **kwargs
    )

