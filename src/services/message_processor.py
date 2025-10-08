"""
Message processor service for orchestrating the message handling flow.

This service coordinates between the database, agent, and other components
to process incoming messages and generate responses.
"""

import logging
from typing import Dict, List, Optional

from src.agents.supervisor_agent import SupervisorAgent
from src.models.database_models import Conversation
from src.services.database_service import DatabaseService

logger = logging.getLogger(__name__)


class MessageProcessor:
    """
    Orchestrates message processing flow.

    Responsibilities:
    1. Manage conversations
    2. Store incoming and outgoing messages
    3. Build conversation context
    4. Coordinate with agent for response generation
    """

    def __init__(self, db_service: DatabaseService, agent: SupervisorAgent):
        """
        Initialize message processor.

        Args:
            db_service: Database service for message storage
            agent: Supervisor agent for message processing
        """
        self.db_service = db_service
        self.agent = agent
        logger.info("MessageProcessor initialized")

    async def process_message(
        self,
        telegram_chat_id: str,
        message_text: str,
        user_context: Optional[dict] = None,
    ) -> str:
        """
        Main entry point: process message and return response.

        This orchestrates the complete flow:
        1. Get or create conversation
        2. Store user message
        3. Build conversation history
        4. Get agent response
        5. Store agent response
        6. Return response text

        Args:
            telegram_chat_id: The Telegram chat ID
            message_text: The user's message text
            user_context: Optional context (user name, etc.)

        Returns:
            str: The agent's response to send back
        """
        logger.info(f"Processing message from chat {telegram_chat_id}")

        try:
            # Get or create conversation
            conversation = await self._get_or_create_conversation(telegram_chat_id)

            # Store incoming user message
            await self._store_user_message(conversation.id, message_text)

            # Build conversation history for agent context
            history = await self._build_conversation_history(conversation.id)

            # Get response from agent
            response_text = await self._get_agent_response(
                message_text, history, user_context
            )

            # Store agent response
            await self._store_agent_response(conversation.id, response_text)

            logger.info(f"Message processed successfully: {response_text[:100]}...")
            return response_text

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # Return friendly error message
            return "I apologize, but I encountered an error processing your message. Please try again."

    async def _get_or_create_conversation(self, chat_id: str) -> Conversation:
        """
        Get or create conversation - delegates to DB service.

        Args:
            chat_id: The Telegram chat ID

        Returns:
            Conversation: The conversation object
        """
        conversation = await self.db_service.get_or_create_conversation(chat_id)
        logger.debug(f"Using conversation ID: {conversation.id}")
        return conversation

    async def _store_user_message(self, conversation_id: int, text: str) -> None:
        """
        Store user message in database.

        Args:
            conversation_id: ID of the conversation
            text: Message text content
        """
        await self.db_service.add_message(
            conversation_id=conversation_id,
            content=text,
            sender="user",
            message_type="text",
        )
        logger.debug("User message stored in database")

    async def _build_conversation_history(
        self, conversation_id: int
    ) -> List[Dict[str, str]]:
        """
        Build conversation history for agent context.

        Retrieves recent messages and converts to agent format.
        Excludes the most recent message (the one we just stored).

        Args:
            conversation_id: ID of the conversation

        Returns:
            List[Dict[str, str]]: Conversation history in agent format
        """
        recent_messages = await self.db_service.get_recent_messages(
            conversation_id=conversation_id, limit=10
        )

        # Convert to agent format, excluding the current message (last one)
        history = []
        for msg in recent_messages[:-1]:
            history.append({"role": msg.sender, "content": msg.content})

        logger.debug(f"Built conversation history with {len(history)} messages")
        return history

    async def _get_agent_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        user_context: Optional[dict],
    ) -> str:
        """
        Get response from agent.

        Args:
            message: The current user message
            history: Previous conversation history
            user_context: Additional user context

        Returns:
            str: The agent's response
        """
        response = await self.agent.process(
            message=message,
            conversation_history=history,
            user_context=user_context or {},
        )
        logger.info(f"Agent generated response: {response[:100]}...")
        return response

    async def _store_agent_response(
        self, conversation_id: int, response: str
    ) -> None:
        """
        Store agent response in database.

        Args:
            conversation_id: ID of the conversation
            response: The agent's response text
        """
        await self.db_service.add_message(
            conversation_id=conversation_id,
            content=response,
            sender="agent",
            message_type="text",
        )
        logger.debug("Agent response stored in database")
