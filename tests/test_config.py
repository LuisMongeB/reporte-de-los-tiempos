import pytest
import os
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError

from src.core.config import (
    BaseConfig, 
    DevelopmentConfig, 
    ProductionConfig, 
    TestConfig,
    load_config,
    ConfigurationError,
    reset_config
)

class TestConfigurationLoading:
    
    def setup_method(self):
        """Reset config before each test"""
        reset_config()
    
    def test_load_development_config(self):
        """Test loading development configuration"""
        with patch.dict(os.environ, {
            'TELEGRAM_BOT_TOKEN': '123456789:ABCdefGHIjklMNOpqrsTUVwxyz',
            'OPENAI_API_KEY': 'sk-test123456789',
            'TELEGRAM_WEBHOOK_SECRET': 'super_secret_webhook_token',
            'TELEGRAM_WEBHOOK_URL': 'https://example.com/webhook'
        }):
            config = load_config("development")
            assert isinstance(config, DevelopmentConfig)
            assert config.debug is True
            assert config.log_level == "DEBUG"
    
    def test_load_production_config_with_required_fields(self):
        """Test loading production config with all required fields"""
        with patch.dict(os.environ, {
            'TELEGRAM_BOT_TOKEN': '123456789:ABCdefGHIjklMNOpqrsTUVwxyz',
            'OPENAI_API_KEY': 'sk-test123456789',
            'TELEGRAM_WEBHOOK_SECRET': 'super_secret_webhook_token',
            'TELEGRAM_WEBHOOK_URL': 'https://example.com/webhook'
        }):
            config = load_config("production")
            assert isinstance(config, ProductionConfig)
            assert config.debug is False
            assert config.telegram_webhook_secret == 'super_secret_webhook_token'
    
    def test_load_config_missing_required_fields(self):
        """Test that missing required fields raise ConfigurationError"""
        # Create a test config class that doesn't read from any env file
        from pydantic_settings import SettingsConfigDict
        
        class TestProductionConfig(ProductionConfig):
            model_config = SettingsConfigDict(
                env_file=None,  # Don't read from any env file
                case_sensitive=False,
                extra="allow"
            )
        
        with patch.dict(os.environ, {}, clear=True):
            # Patch the config class in load_config
            with patch('src.core.config.CONFIG_CLASSES', {'production': TestProductionConfig}):
                with pytest.raises(ConfigurationError):
                    load_config("production")
    
    def test_invalid_environment_name(self):
        """Test that invalid environment names raise ConfigurationError"""
        with pytest.raises(ConfigurationError) as exc_info:
            load_config("invalid_environment")
        
        assert "Unknown environment" in str(exc_info.value)
    
    def test_config_summary_hides_sensitive_data(self):
        """Test that configuration summary hides sensitive information"""
        config = load_config("test")
        summary = config.get_summary()
        
        assert summary['telegram_bot_token'] == "***HIDDEN***"
        assert summary['openai_api_key'] == "***HIDDEN***"
        assert summary['app_name'] == config.app_name
    
    def test_explicit_config_class(self):
        """Test loading configuration with explicit config class"""
        config = load_config(config_class=TestConfig)
        assert isinstance(config, TestConfig)
        assert config.database_url == "sqlite+aiosqlite:///:memory:"