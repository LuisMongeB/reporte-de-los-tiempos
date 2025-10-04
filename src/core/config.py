import logging
import os
from pathlib import Path
from typing import Literal, Optional, Type

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """Base configuration class with common settings"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # App Settings
    app_name: str = Field(default="Telegram AI Agent", description="Application name")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    # Database Settings
    database_url: str = Field(
        default="sqlite+aiosqlite:///./telegram_agent.db",
        description="Database connection URL",
    )
    database_echo: bool = Field(default=False, description="Echo SQL queries")

    # Telegram Settings
    telegram_bot_token: str = Field(
        ..., description="Telegram bot token from BotFather"
    )
    telegram_webhook_secret: Optional[str] = Field(
        default=None, description="Secret token for webhook validation"
    )
    telegram_webhook_url: Optional[str] = Field(
        default=None, description="Full webhook URL (will be set dynamically)"
    )
    telegram_webhook_path: str = Field(
        default="/webhook/telegram", description="Webhook endpoint path"
    )
    telegram_api_base_url: str = Field(
        default="https://api.telegram.org", description="Telegram Bot API base URL"
    )
    telegram_connect_timeout: int = Field(
        default=30, description="Connection timeout for Telegram API calls in seconds"
    )
    telegram_read_timeout: int = Field(
        default=30, description="Read timeout for Telegram API calls in seconds"
    )

    # OpenAI Settings
    openai_api_key: Optional[str] = Field(default=None)
    llm_model: str = Field("gpt-5-nano")
    llm_temperature: float = Field(default=0.7)
    llm_max_tokens: int = Field(default=1000)

    # Agent Settings
    conversation_history_limit: int = Field(
        default=10, description="Number of recent messages to include in agent context"
    )

    # File Storage
    temp_dir: Path = Field(default=Path("./temp"), description="Temporary file storage")
    max_file_size_mb: int = Field(default=25, description="Maximum file size in MB")

    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    @field_validator("telegram_bot_token")
    @classmethod
    def validate_telegram_token(cls, v: str) -> str:
        if not v or len(v) < 10:
            raise ValueError("Telegram bot token must be provided and valid")
        return v

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        if not v or not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v

    @field_validator("temp_dir")
    @classmethod
    def create_temp_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v

    def get_summary(self) -> dict:
        """Get a summary of configuration (excluding sensitive data)"""
        sensitive_fields = {
            "telegram_bot_token",
            "openai_api_key",
            "telegram_webhook_secret",
        }

        summary = {}
        for field_name, field_value in self.model_dump().items():
            if field_name in sensitive_fields:
                summary[field_name] = "***HIDDEN***" if field_value else "NOT SET"
            else:
                summary[field_name] = field_value

        return summary


class DevelopmentConfig(BaseConfig):
    """Development environment configuration"""

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    database_echo: bool = True

    # Development-specific settings
    ngrok_auth_token: Optional[str] = Field(
        default=None, description="ngrok auth token for tunneling"
    )
    auto_setup_webhook: bool = Field(
        default=True, description="Automatically set up webhook via ngrok"
    )

    # More permissive settings for development
    llm_max_tokens: int = 500


class ProductionConfig(BaseConfig):
    """Production environment configuration"""

    model_config = SettingsConfigDict(
        env_file=".env.prod",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    database_echo: bool = False

    # Production-specific validations
    telegram_webhook_secret: str = Field(
        ..., description="Webhook secret is required in production"
    )
    telegram_webhook_url: str = Field(
        ..., description="Webhook URL is required in production"
    )

    # Stricter settings for production
    host: str = "127.0.0.1"

    @field_validator("telegram_webhook_url")
    @classmethod
    def validate_webhook_url(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("Webhook URL must use HTTPS in production")
        return v


class TestConfig(BaseConfig):
    """Test environment configuration"""

    model_config = SettingsConfigDict(
        env_file=".env.test",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    database_url: str = "sqlite+aiosqlite:///:memory:"

    # Override required fields for testing
    telegram_bot_token: str = "123456789ABCdefGHIjklMNOpqrsTUVwxyz"
    openai_api_key: str = "sk-test-key-for-testing-only"


# Configuration Error Class
class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails"""

    pass


# Configuration mapping
CONFIG_CLASSES = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "test": TestConfig,
}


def load_config(
    environment: Optional[str] = None, config_class: Optional[Type[BaseConfig]] = None
) -> BaseConfig:
    """Load configuration based on environment or explicit config class."""

    if config_class:
        selected_config_class = config_class
        source = f"explicit class {config_class.__name__}"
    elif environment:
        selected_config_class = CONFIG_CLASSES.get(environment)
        if not selected_config_class:
            raise ConfigurationError(f"Unknown environment: {environment}")
        source = f"environment parameter '{environment}'"
    else:
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        selected_config_class = CONFIG_CLASSES.get(env_name, DevelopmentConfig)
        source = f"ENVIRONMENT variable '{env_name}'"

    try:
        config = selected_config_class()
        logging.info(f"Configuration loaded successfully from {source}")
        return config

    except ValidationError as e:
        error_msg = f"Configuration validation failed when loading from {source}"
        logging.error(f"{error_msg}: {e}")
        raise ConfigurationError(f"{error_msg}. Details: {e}")

    except Exception as e:
        error_msg = f"Unexpected error loading configuration from {source}"
        logging.error(f"{error_msg}: {e}")
        raise ConfigurationError(f"{error_msg}: {e}")


# Global configuration instance
_config: Optional[BaseConfig] = None


def get_config() -> BaseConfig:
    """Get the global configuration instance, loading it if necessary"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config():
    """Reset the global configuration (useful for testing)"""
    global _config
    _config = None


def validate_config_completeness(config: BaseConfig) -> None:
    """Validate that all required configuration is present and valid."""
    errors = []

    # Check file system permissions
    try:
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        test_file = config.temp_dir / "test_write.txt"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        errors.append(f"Cannot write to temp directory {config.temp_dir}: {e}")

    # Validate database path (for SQLite)
    if config.database_url.startswith("sqlite"):
        db_path = config.database_url.split("///")[-1]
        if db_path != ":memory:":
            db_dir = Path(db_path).parent
            if not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create database directory {db_dir}: {e}")

    if errors:
        raise ConfigurationError(
            "Configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
