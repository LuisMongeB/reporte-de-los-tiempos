"""
Telegram webhook endpoint for receiving updates.
"""

from fastapi import APIRouter, Request, HTTPException, status, Header
from typing import Optional
import logging
import hmac
import hashlib

from src.api.models import TelegramUpdate
from src.core.config import get_config

logger = logging.getLogger(__name__)
settings = get_config()

router = APIRouter(prefix="/webhook", tags=["telegram"])


def validate_telegram_webhook(
    body: bytes,
    secret_token: Optional[str] = None
) -> bool:
    """
    Validate that the request came from Telegram using the secret token.

    Args:
        body: Raw request body as bytes
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


@router.post("/telegram")
async def telegram_webhook(
    update: TelegramUpdate,
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None)
):
    """
    Receive incoming updates from Telegram.

    This endpoint MUST respond quickly (< 5 seconds) to avoid Telegram timeouts.
    Message processing happens asynchronously after the response is sent.

    Args:
        update: The Telegram update object
        request: FastAPI request object (for raw body access)
        x_telegram_bot_api_secret_token: Secret token header from Telegram

    Returns:
        dict: Simple acknowledgment
    """
    logger.info(f"Received Telegram update: {update.update_id}")

    # Validate the request came from Telegram
    body = await request.body()
    if not validate_telegram_webhook(body, x_telegram_bot_api_secret_token):
        logger.error("Webhook validation failed")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid webhook secret"
        )

    # For now, just log the message and return OK
    # Processing will be added in Phase 5
    if update.message and update.message.text:
        logger.info(
            f"Message from user {update.message.from_user.id}: "
            f"{update.message.text[:50]}..."
        )

    # CRITICAL: Return 200 OK immediately
    return {"ok": True}
