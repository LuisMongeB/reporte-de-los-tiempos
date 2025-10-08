"""
Telegram webhook endpoint for receiving updates.
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, status

from src.agents.supervisor_agent import SupervisorAgent
from src.api.models import TelegramUpdate
from src.core.config import get_config
from src.core.database import DatabaseManager
from src.services.database_service import DatabaseService
from src.services.message_processor import MessageProcessor
from src.services.telegram_service import TelegramService

logger = logging.getLogger(__name__)
settings = get_config()

router = APIRouter(prefix="/webhook", tags=["telegram"])

# ============================================================================
# SERVICE INITIALIZATION (Singleton Pattern)
# ============================================================================
# These are initialized once when the first webhook arrives, then reused.
# This is efficient for Raspberry Pi - avoids recreating services per request.

_db_manager: Optional[DatabaseManager] = None
_db_service: Optional[DatabaseService] = None
_telegram_service: Optional[TelegramService] = None
_agent: Optional[SupervisorAgent] = None
_message_processor: Optional[MessageProcessor] = None


def get_message_processor() -> MessageProcessor:
    """
    Get or create the message processor instance (lazy initialization).

    This ensures services are only created when the first message arrives,
    not at app startup.

    Returns:
        MessageProcessor: Initialized message processor
    """
    global _db_manager, _db_service, _telegram_service, _agent, _message_processor

    if _message_processor is None:
        logger.info("Initializing message processing services...")

        # Initialize database service
        _db_manager = DatabaseManager(settings.database_url)
        _db_service = DatabaseService(_db_manager)

        # Initialize Telegram service
        _telegram_service = TelegramService()

        # Initialize agent
        _agent = SupervisorAgent()

        # Initialize message processor
        _message_processor = MessageProcessor(_db_service, _agent)

        logger.info("Message processing services initialized successfully")

    return _message_processor


def get_telegram_service() -> TelegramService:
    """
    Get the Telegram service instance.

    Ensures initialization by calling get_message_processor first.

    Returns:
        TelegramService: Initialized telegram service
    """
    get_message_processor()  # Ensures all services are initialized
    return _telegram_service


# ============================================================================
# WEBHOOK VALIDATION
# ============================================================================


def validate_telegram_webhook(secret_token: Optional[str] = None) -> bool:
    """
    Validate that the request came from Telegram using the secret token.

    Args:
        secret_token: Secret token from X-Telegram-Bot-Api-Secret-Token header

    Returns:
        bool: True if valid, False otherwise
    """
    # If no webhook secret is configured, skip validation (dev mode)
    if not settings.telegram_webhook_secret:
        logger.warning("No webhook secret configured - skipping validation")
        return True

    # Check if the secret token matches
    if secret_token != settings.telegram_webhook_secret:
        logger.warning("Invalid webhook secret token received")
        return False

    return True


# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================


async def process_and_respond(
    update: TelegramUpdate,
    processor: MessageProcessor,
    telegram_service: TelegramService,
):
    """
    Process message and send response (runs in background after webhook returns).

    This function handles the complete flow:
    1. Extract message details
    2. Build user context
    3. Send typing indicator
    4. Process message through agent
    5. Send response back to user

    Runs asynchronously after the webhook has returned 200 OK to Telegram.

    Args:
        update: The Telegram update object
        processor: Message processor instance
        telegram_service: Telegram service for sending responses
    """
    try:
        # Extract message details
        message = update.message
        if not message or not message.text:
            logger.warning("Update has no text message, skipping")
            return

        chat_id = message.chat.id
        message_text = message.text

        # Build user context from Telegram data
        user_context = {}
        if message.from_user:
            user_context["user_id"] = message.from_user.id
            user_context["is_bot"] = message.from_user.is_bot
            user_context["user_name"] = message.from_user.first_name
            if message.from_user.last_name:
                user_context["user_name"] += f" {message.from_user.last_name}"
            if message.from_user.username:
                user_context["username"] = message.from_user.username

        logger.info(
            f"Processing message from chat {chat_id} "
            f"(user_id: {user_context.get('user_id', 'unknown')}): "
            f"{message_text[:50]}{'...' if len(message_text) > 50 else ''}"
        )

        # Send typing indicator to show bot is processing
        await telegram_service.send_typing_action(chat_id)

        # Process message through agent
        response_text = await processor.process_message(
            telegram_chat_id=str(chat_id),
            message_text=message_text,
            user_context=user_context,
        )

        logger.info(
            f"Agent generated response for chat {chat_id}: "
            f"{response_text[:100]}{'...' if len(response_text) > 100 else ''}"
        )

        # Send response back to user
        result = await telegram_service.send_message(chat_id=chat_id, text=response_text)

        if result.success:
            logger.info(
                f"Response sent successfully to chat {chat_id}, "
                f"message_id: {result.message.message_id}"
            )
        else:
            logger.error(
                f"Failed to send response to chat {chat_id}: "
                f"{result.error} - {result.error_details}"
            )

    except Exception as e:
        logger.error(f"Error in background processing: {e}", exc_info=True)
        # Attempt to send error message to user
        try:
            await telegram_service.send_message(
                chat_id=chat_id,
                text="I apologize, but I encountered an error. Please try again.",
            )
            logger.info(f"Error notification sent to chat {chat_id}")
        except Exception as send_error:
            logger.error(f"Failed to send error message to chat {chat_id}: {send_error}")


# ============================================================================
# WEBHOOK ENDPOINT
# ============================================================================


@router.post("/telegram")
async def telegram_webhook(
    update: TelegramUpdate,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None),
):
    """
    Receive incoming updates from Telegram.

    This endpoint MUST respond quickly (< 5 seconds) to avoid Telegram timeouts.
    Message processing happens asynchronously in the background.

    Flow:
    1. Validate webhook secret
    2. Schedule background processing
    3. Return 200 OK immediately to Telegram
    4. Background: Process message and send response

    Args:
        update: The Telegram update object
        background_tasks: FastAPI background tasks for async processing
        x_telegram_bot_api_secret_token: Secret token header from Telegram

    Returns:
        dict: Simple acknowledgment

    Raises:
        HTTPException: If webhook validation fails
    """
    logger.info(f"Received Telegram update: {update.update_id}")

    # Validate the request came from Telegram
    if not validate_telegram_webhook(x_telegram_bot_api_secret_token):
        logger.error("Webhook validation failed")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid webhook secret"
        )

    # Get services (lazy initialization on first call)
    processor = get_message_processor()
    telegram_service = get_telegram_service()

    # Schedule background processing
    if update.message and update.message.text:
        background_tasks.add_task(process_and_respond, update, processor, telegram_service)
        logger.info(
            f"Message processing scheduled in background for update {update.update_id}"
        )
    else:
        logger.warning(
            f"Update {update.update_id} contains no text message, ignoring"
        )

    # CRITICAL: Return 200 OK immediately
    return {"ok": True}
