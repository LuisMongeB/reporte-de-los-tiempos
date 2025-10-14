"""
Database service layer for conversation and message management.

This service provides a clean interface for all database operations,
abstracting SQLAlchemy details from the rest of the application.
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import DatabaseManager
from src.models.database_models import Conversation, Message

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Service layer for database operations.

    Handles all interactions with the database for conversations and messages.
    Provides a clean interface for the rest of the application.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the database service.

        Args:
            db_manager: DatabaseManager instance for database operations
        """
        self.db_manager = db_manager
        logger.info("DatabaseService initialized")

    async def get_or_create_conversation(self, telegram_chat_id: str) -> Conversation:
        """
        Get an existing conversation or create a new one.

        This is the main entry point for conversation management. When a message
        arrives from Telegram, we look up the conversation by chat_id. If it
        doesn't exist, we create it.

        Args:
            telegram_chat_id: The Telegram chat ID (as string for consistency)

        Returns:
            Conversation: The conversation object (existing or newly created)

        Raises:
            Exception: If database operation fails
        """
        try:
            async with self.db_manager.get_session() as session:
                # Try to find existing conversation
                result = await session.execute(
                    select(Conversation).where(
                        Conversation.telegram_chat_id == telegram_chat_id
                    )
                )
                conversation = result.scalar_one_or_none()

                # Create new conversation if not found
                if not conversation:
                    logger.info(
                        f"Creating new conversation for chat_id: {telegram_chat_id}"
                    )
                    conversation = Conversation(
                        telegram_chat_id=telegram_chat_id, status="active"
                    )
                    session.add(conversation)
                    await session.commit()
                    await session.refresh(conversation)
                    logger.info(f"Created conversation with id: {conversation.id}")
                else:
                    logger.debug(f"Found existing conversation: {conversation.id}")

                return conversation

        except Exception as e:
            logger.error(f"Error getting/creating conversation: {e}", exc_info=True)
            raise

    async def add_message(
        self,
        conversation_id: int,
        content: str,
        sender: str,
        message_type: str = "text",
        telegram_message_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Message:
        """
        Add a new message to a conversation.

        This stores both user messages and agent responses. The sender field
        distinguishes between them.

        Our Message model includes the following fields:
        - conversation_id: Foreign key to conversation
        - telegram_message_id: Telegram's unique message ID (null for agent messages)
        - content: The message text content
        - sender: "user" or "agent" to distinguish message origin
        - message_type: Message type ("text", "audio", "voice", "system")
        - metadata: JSON field for additional message metadata
        - timestamp: Auto-generated timestamp
        - User info fields: telegram_user_id, telegram_username, telegram_first_name, telegram_last_name
        - File metadata: file_id, file_size, duration (for audio/voice)

        Args:
            conversation_id: ID of the conversation this message belongs to
            content: The message text content
            sender: Either "user" or "agent" (or specific agent name later)
            message_type: Type of message (text, audio, voice, system) - default "text"
            telegram_message_id: Optional Telegram message ID (for user messages)
            metadata: Optional dictionary with additional message metadata

        Returns:
            Message: The created message object

        Raises:
            Exception: If database operation fails
        """
        try:
            async with self.db_manager.get_session() as session:
                message = Message(
                    conversation_id=conversation_id,
                    content=content,
                    sender=sender,
                    message_type=message_type,
                    telegram_message_id=telegram_message_id,
                    message_metadata=metadata or {},
                )
                session.add(message)
                await session.commit()
                await session.refresh(message)

                logger.debug(
                    f"Added message to conversation {conversation_id}: "
                    f"{content[:50]}... (sender: {sender})"
                )

                return message

        except Exception as e:
            logger.error(f"Error adding message: {e}", exc_info=True)
            raise

    async def get_recent_messages(
        self, conversation_id: int, limit: int = 10
    ) -> List[Message]:
        """
        Retrieve the most recent messages from a conversation.

        This is used to provide context to the agent. The limit parameter
        controls how much history is included.

        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of messages to retrieve (default 10)

        Returns:
            List[Message]: List of messages, ordered oldest first (chronological)

        Raises:
            Exception: If database operation fails
        """
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(desc(Message.timestamp), desc(Message.id))
                    .limit(limit)
                )
                messages = result.scalars().all()

                logger.debug(
                    f"Retrieved {len(messages)} messages for conversation {conversation_id}"
                )

                # Return in chronological order (oldest first) for agent context
                return list(reversed(messages))

        except Exception as e:
            logger.error(f"Error retrieving messages: {e}", exc_info=True)
            raise

    async def get_conversation_by_id(
        self, conversation_id: int
    ) -> Optional[Conversation]:
        """
        Retrieve a conversation by its ID.

        Args:
            conversation_id: The conversation ID

        Returns:
            Optional[Conversation]: The conversation if found, None otherwise
        """
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(Conversation).where(Conversation.id == conversation_id)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error retrieving conversation: {e}", exc_info=True)
            raise
