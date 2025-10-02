# tests/conftest.py
import sys
from pathlib import Path

import pytest

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_telegram_update():
    """
    Mock Telegram update payload for testing webhook endpoints.

    This fixture provides a complete, valid Telegram update structure
    that matches Telegram's actual API format. Can be used across
    multiple test files for webhook and message processing tests.
    """
    return {
        "update_id": 123456789,
        "message": {
            "message_id": 1,
            "from": {
                "id": 987654321,
                "is_bot": False,
                "first_name": "Test",
                "last_name": "User",
                "username": "testuser",
                "language_code": "en"
            },
            "chat": {
                "id": 987654321,
                "first_name": "Test",
                "last_name": "User",
                "username": "testuser",
                "type": "private"
            },
            "date": 1234567890,
            "text": "Hello, bot!"
        }
    }


@pytest.fixture
def mock_telegram_update_no_text():
    """
    Mock Telegram update without text content.

    Useful for testing how the system handles updates that don't
    contain text messages (e.g., stickers, or future file uploads).
    """
    return {
        "update_id": 123456790,
        "message": {
            "message_id": 2,
            "from": {
                "id": 987654321,
                "is_bot": False,
                "first_name": "Test",
                "username": "testuser"
            },
            "chat": {
                "id": 987654321,
                "type": "private"
            },
            "date": 1234567891
            # No text field
        }
    }


@pytest.fixture
def mock_telegram_update_no_message():
    """
    Mock Telegram update that doesn't contain a message.

    Telegram can send various update types (edited_message, channel_post, etc.).
    This tests handling of non-message updates.
    """
    return {
        "update_id": 123456791
        # No message field
    }