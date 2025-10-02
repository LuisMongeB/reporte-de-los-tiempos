"""
Pydantic models for API requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Generic, Optional, TypeVar, Union
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field


# Type variable for generic responses
T = TypeVar('T')

# Export all models
__all__ = [
    'ResponseStatus',
    'BaseResponse',
    'ErrorResponse',
    'SuccessResponse',
    'HealthStatus',
    'HealthResponse',
    'DatabaseStatus',
    'ConfigSummary',
    'DetailedHealthResponse',
    'TelegramUser',
    'TelegramChat',
    'TelegramMessage',
    'TelegramUpdate'
]


class ResponseStatus(str, Enum):
    """Response status enum for API responses"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    
    def __str__(self) -> str:
        return self.value


class BaseResponse(BaseModel):
    """
    Base response model with common fields for all API responses.
    Can be used directly or extended by other response models.
    """
    timestamp: str = Field(
        description="ISO timestamp when the response was generated",
        json_schema_extra={"example": "2024-01-15T14:30:00+01:00"}
    )
    status: ResponseStatus = Field(
        description="Response status indicator"
    )
    
    def __init__(self, **data):
        # Auto-generate timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now(ZoneInfo("Europe/Rome")).isoformat()
        super().__init__(**data)


class ErrorResponse(BaseResponse):
    """Response model for error cases"""
    status: ResponseStatus = Field(default=ResponseStatus.ERROR, description="Always 'error' for error responses")
    error: str = Field(description="Main error message")
    detail: Optional[str] = Field(default=None, description="Optional detailed error description")
    request_id: Optional[str] = Field(default=None, description="Optional request ID for tracing")
    
    def __init__(self, **data):
        # Force status to always be ERROR
        data['status'] = ResponseStatus.ERROR
        super().__init__(**data)


class SuccessResponse(BaseResponse, Generic[T]):
    """Generic response model for successful operations with data"""
    status: ResponseStatus = Field(default=ResponseStatus.SUCCESS, description="Always 'success' for successful responses")
    data: T = Field(description="Response data payload")
    
    def __init__(self, **data):
        # Force status to always be SUCCESS
        data['status'] = ResponseStatus.SUCCESS
        super().__init__(**data)


# Health Check Models

class HealthStatus(str, Enum):
    """Health status enum for health check responses"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    
    def __str__(self) -> str:
        return self.value


class HealthResponse(BaseResponse):
    """Basic health check response with status and timestamp"""
    health_status: HealthStatus = Field(description="Overall health status of the service")
    
    def __init__(self, **data):
        # Auto-set ResponseStatus based on health status
        if 'health_status' in data:
            if data['health_status'] == HealthStatus.HEALTHY:
                data['status'] = ResponseStatus.SUCCESS
            elif data['health_status'] == HealthStatus.DEGRADED:
                data['status'] = ResponseStatus.PENDING
            else:  # UNHEALTHY
                data['status'] = ResponseStatus.ERROR
        super().__init__(**data)


class DatabaseStatus(BaseModel):
    """Database connection status and performance metrics"""
    is_connected: bool = Field(description="Whether database connection is active")
    response_time_ms: Optional[float] = Field(
        default=None,
        description="Database response time in milliseconds",
        ge=0
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if database connection failed"
    )
    
    @property
    def status(self) -> HealthStatus:
        """Derive health status from connection state and response time"""
        if not self.is_connected:
            return HealthStatus.UNHEALTHY
        elif self.response_time_ms and self.response_time_ms > 1000:  # > 1 second
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


class ConfigSummary(BaseModel):
    """Safe configuration summary without sensitive data"""
    app_name: str = Field(description="Application name")
    debug: bool = Field(description="Debug mode status")
    log_level: str = Field(description="Current logging level")
    database_type: str = Field(description="Type of database (e.g., 'sqlite', 'postgresql')")
    telegram_configured: bool = Field(description="Whether Telegram bot token is configured")
    openai_configured: bool = Field(description="Whether OpenAI API key is configured")
    webhook_configured: bool = Field(description="Whether webhook is configured")
    temp_dir: str = Field(description="Temporary directory path")
    max_file_size_mb: int = Field(description="Maximum file size in MB")
    
    @classmethod
    def from_config(cls, config) -> 'ConfigSummary':
        """Create ConfigSummary from a configuration object"""
        # Extract database type from URL
        db_url = config.database_url.lower()
        if 'sqlite' in db_url:
            db_type = 'sqlite'
        elif 'postgresql' in db_url or 'postgres' in db_url:
            db_type = 'postgresql'
        elif 'mysql' in db_url:
            db_type = 'mysql'
        else:
            db_type = 'unknown'
        
        return cls(
            app_name=config.app_name,
            debug=config.debug,
            log_level=config.log_level,
            database_type=db_type,
            telegram_configured=bool(config.telegram_bot_token),
            openai_configured=bool(config.openai_api_key),
            webhook_configured=bool(getattr(config, 'telegram_webhook_url', None)),
            temp_dir=str(config.temp_dir),
            max_file_size_mb=config.max_file_size_mb
        )


class DetailedHealthResponse(BaseResponse):
    """Comprehensive health check response with all system information"""
    health_status: HealthStatus = Field(description="Overall health status of the service")
    database: DatabaseStatus = Field(description="Database connection status and metrics")
    config: ConfigSummary = Field(description="Safe configuration summary")
    uptime_seconds: Optional[float] = Field(
        default=None,
        description="Service uptime in seconds",
        ge=0
    )
    version: Optional[str] = Field(
        default=None,
        description="Application version"
    )
    
    def __init__(self, **data):
        # Auto-determine overall health status if not provided
        if 'health_status' not in data and 'database' in data:
            db_status = data['database']
            if isinstance(db_status, DatabaseStatus):
                data['health_status'] = db_status.status
            elif isinstance(db_status, dict):
                # Create DatabaseStatus from dict to get status
                db_obj = DatabaseStatus(**db_status)
                data['health_status'] = db_obj.status
        
        # Auto-set ResponseStatus based on health status
        if 'health_status' in data:
            if data['health_status'] == HealthStatus.HEALTHY:
                data['status'] = ResponseStatus.SUCCESS
            elif data['health_status'] == HealthStatus.DEGRADED:
                data['status'] = ResponseStatus.PENDING
            else:  # UNHEALTHY
                data['status'] = ResponseStatus.ERROR

        super().__init__(**data)


# Telegram Webhook Models

class TelegramUser(BaseModel):
    """Represents a Telegram user"""
    id: int = Field(..., description="Unique identifier for this user")
    is_bot: bool = Field(False, description="True if this user is a bot")
    first_name: str = Field(..., description="User's first name")
    last_name: Optional[str] = Field(None, description="User's last name")
    username: Optional[str] = Field(None, description="User's username")
    language_code: Optional[str] = Field(None, description="IETF language tag")


class TelegramChat(BaseModel):
    """Represents a Telegram chat"""
    id: int = Field(..., description="Unique identifier for this chat")
    type: str = Field(..., description="Type of chat: private, group, supergroup, channel")
    title: Optional[str] = Field(None, description="Title for supergroups, channels, group chats")
    username: Optional[str] = Field(None, description="Username for private chats, supergroups, channels")
    first_name: Optional[str] = Field(None, description="First name of the other party in private chat")
    last_name: Optional[str] = Field(None, description="Last name of the other party in private chat")


class TelegramMessage(BaseModel):
    """Represents a Telegram message"""
    message_id: int = Field(..., description="Unique message identifier")
    from_user: Optional[TelegramUser] = Field(None, alias="from", description="Sender of the message")
    chat: TelegramChat = Field(..., description="Conversation the message belongs to")
    date: int = Field(..., description="Date the message was sent in Unix time")

    # Text content
    text: Optional[str] = Field(None, description="For text messages, the actual UTF-8 text of the message")

    # Media content (Phase 2+ - not processed in MVP)
    photo: Optional[list] = Field(None, description="Available sizes of the photo")
    document: Optional[dict] = Field(None, description="Information about the document file")
    audio: Optional[dict] = Field(None, description="Information about the audio file")
    voice: Optional[dict] = Field(None, description="Information about the voice message")
    video: Optional[dict] = Field(None, description="Information about the video file")

    # Caption for media
    caption: Optional[str] = Field(None, description="Caption for the media, 0-1024 characters")

    class Config:
        populate_by_name = True  # Allows both 'from' and 'from_user' as field names


class TelegramUpdate(BaseModel):
    """Represents an incoming Telegram update"""
    update_id: int = Field(..., description="Unique identifier for this update")
    message: Optional[TelegramMessage] = Field(None, description="New incoming message")

    # Future: edited_message, channel_post, etc. can be added here
