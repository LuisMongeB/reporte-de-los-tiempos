"""
Telegram service for sending messages back to users.

This service handles all interactions with the Telegram Bot API,
primarily sending messages but designed to be extended with
editing, deleting, and other operations.
"""

import logging
from typing import Optional

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from src.core.config import get_config
from src.services.telegram_models import TelegramSendResult

logger = logging.getLogger(__name__)
settings = get_config()


class TelegramService:
    """
    Service for interacting with Telegram Bot API.

    Handles sending messages back to users.
    Designed to be extended with editing, typing indicators, and other features.
    """

    def __init__(self):
        """Initialize Telegram bot."""
        self.bot = Bot(token=settings.telegram_bot_token)
        logger.info("TelegramService initialized")

    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: Optional[str] = ParseMode.MARKDOWN_V2,
    ) -> TelegramSendResult:
        """
        Send a text message to a Telegram chat.

        Args:
            chat_id: The Telegram chat ID to send to
            text: The message text to send
            parse_mode: Parse mode for formatting (default: MarkdownV2)

        Returns:
            TelegramSendResult: Result object containing:
                - success: bool indicating if send was successful
                - message: Telegram Message object (for future edits) if successful
                - error: Error code/type if failed
                - error_details: Detailed error message if failed

        Example:
            result = await service.send_message(12345, "Hello *world*!")
            if result.success:
                message_id = result.message.message_id
                print(f"Sent with ID: {message_id}")
            else:
                print(f"Failed: {result.error}")
        """
        try:
            message = await self.bot.send_message(
                chat_id=chat_id, text=text, parse_mode=parse_mode
            )

            logger.info(
                f"Message sent successfully to chat {chat_id}, "
                f"message_id: {message.message_id}"
            )

            return TelegramSendResult(
                success=True, message=message, error=None, error_details=None
            )

        except TelegramError as e:
            # Telegram-specific errors (BadRequest, Forbidden, etc.)
            error_code = type(e).__name__
            logger.error(
                f"Telegram API error sending to chat {chat_id}: "
                f"{error_code} - {str(e)}",
                exc_info=True,
            )

            return TelegramSendResult(
                success=False, message=None, error=error_code, error_details=str(e)
            )

        except Exception as e:
            # Unexpected errors
            error_type = type(e).__name__
            logger.error(
                f"Unexpected error sending message to chat {chat_id}: "
                f"{error_type} - {str(e)}",
                exc_info=True,
            )

            return TelegramSendResult(
                success=False, message=None, error=error_type, error_details=str(e)
            )

    async def send_typing_action(self, chat_id: int) -> bool:
        """
        Send typing indicator to show the bot is processing.
        
        This shows "Bot is typing..." to the user.
        The indicator automatically disappears after 5 seconds or when a message is sent.
        
        Args:
            chat_id: The Telegram chat ID
                
        Returns:
            bool: True if sent successfully, False otherwise
            
        Example:
            await service.send_typing_action(12345)
            # User sees "Bot is typing..."
        """
        try:
            await self.bot.send_chat_action(chat_id=chat_id, action="typing")
            logger.debug(f"Typing indicator sent to chat {chat_id}")
            return True
            
        except TelegramError as e:
            # Don't raise errors for typing indicator - it's not critical
            logger.warning(
                f"Failed to send typing indicator to {chat_id}: {e}"
            )
            return False
            
        except Exception as e:
            logger.warning(
                f"Unexpected error sending typing indicator to {chat_id}: {e}"
            )
            return False
