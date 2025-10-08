"""
Data models for Telegram service responses.

These models provide type-safe response structures for Telegram operations,
making it easier to handle results and errors consistently.
"""

from typing import Optional

from pydantic import BaseModel, Field
from telegram import Message


class TelegramSendResult(BaseModel):
    """
    Result of sending a Telegram message.
    
    Attributes:
        success: Whether the message was sent successfully
        message: The Telegram Message object if successful (contains message_id for future edits)
        error: Error type/code if the send failed
        error_details: Detailed error message for debugging
    """

    success: bool = Field(..., description="Whether the message was sent successfully")
    message: Optional[Message] = Field(
        None, description="Telegram Message object if successful"
    )
    error: Optional[str] = Field(None, description="Error type/code if failed")
    error_details: Optional[str] = Field(
        None, description="Detailed error message"
    )

    model_config = {
        "arbitrary_types_allowed": True  # Required for Telegram Message object
    }
