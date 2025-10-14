"""
Database models for the Telegram AI Agent.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.core.database import Base


class TimestampMixin:
    """Mixin for automatic timestamp management"""
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Conversation(Base, TimestampMixin):
    """Conversation table - represents a chat session with a Telegram user"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_chat_id = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="active")  # "active", "completed"
    
    # Relationships
    messages = relationship("Message", back_populates="conversation")
    agent_states = relationship("AgentState", back_populates="conversation")
    
    # Indexes
    __table_args__ = (
        Index('ix_conversations_telegram_chat_id', 'telegram_chat_id'),
    )


class Message(Base):
    """Messages table - stores all messages in conversations"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    telegram_message_id = Column(String(255), nullable=True)  # Telegram's message ID (null for agent messages)
    content = Column(Text, nullable=True)
    sender = Column(String(50), nullable=False)  # "user" or "agent"
    message_type = Column(String(50), nullable=False)  # "text", "audio", "voice", "system"
    message_metadata = Column(JSON, nullable=True)  # Additional message metadata (renamed from 'metadata' to avoid SQLAlchemy conflict)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # User information (basic Telegram metadata)
    telegram_user_id = Column(String(255), nullable=True)
    telegram_username = Column(String(255), nullable=True)
    telegram_first_name = Column(String(255), nullable=True)
    telegram_last_name = Column(String(255), nullable=True)
    
    # File metadata (for audio/voice messages)
    file_id = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=True)
    duration = Column(Integer, nullable=True)  # seconds for audio/voice
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index('ix_messages_conversation_id', 'conversation_id'),
        Index('ix_messages_timestamp', 'timestamp'),
        Index('ix_messages_telegram_message_id', 'telegram_message_id'),
    )


class AgentState(Base):
    """Agent states table - stores LangGraph state data"""
    __tablename__ = "agent_states"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    state_data = Column(JSON, nullable=False)  # LangGraph state as JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="agent_states")
    
    # Indexes
    __table_args__ = (
        Index('ix_agent_states_conversation_id', 'conversation_id'),
        Index('ix_agent_states_created_at', 'created_at'),
    )