"""
Agent Module Initialization

This module automatically registers all specialized agents with the AgentRegistry
when the agents module is imported. This ensures agents are discoverable for
dynamic routing.

The registry uses a singleton pattern, so agents are registered once globally
and can be accessed from anywhere in the application.
"""

import logging

logger = logging.getLogger(__name__)

# Import registry first
from src.agents.agent_registry import AgentRegistry

# Import specialized agents
from src.agents.url_extractor_agent import URLExtractorAgent
from src.agents.pdf_processor_agent import PDFProcessorAgent
from src.agents.notion_agent import NotionAgent

# Agent configuration for registration
AGENT_CONFIGS = [
    {
        "agent_id": "url_extractor",
        "agent_class": URLExtractorAgent,
        "prompt_file": "specialized/url_extractor_agent.yaml",
        "required_settings": ["tavily_api_key", "openai_api_key"],
        "categories": ["content_extraction", "web"]
    },
    {
        "agent_id": "pdf_processor",
        "agent_class": PDFProcessorAgent,
        "prompt_file": "specialized/pdf_processor_agent.yaml",
        "required_settings": ["openai_api_key"],  # Vision LLM uses OpenAI
        "categories": ["content_extraction", "document_processing"]
    },
    {
        "agent_id": "notion_agent",
        "agent_class": NotionAgent,
        "prompt_file": "specialized/notion_agent.yaml",
        "required_settings": ["notion_api_key", "notion_default_parent_page_id"],
        "categories": ["storage", "integration"]
    },
]


def register_agents():
    """
    Register all specialized agents with the AgentRegistry.

    This function is called automatically when the module is imported.
    Agents that fail to register (e.g., due to missing dependencies) will log
    a warning but won't prevent other agents from registering.
    """
    registry = AgentRegistry.get_instance()

    for config in AGENT_CONFIGS:
        try:
            registry.register(**config)
            logger.info(f" Registered agent: {config['agent_id']}")
        except Exception as e:
            logger.warning(
                f" Could not register agent {config['agent_id']}: {e}. "
                f"Agent will be unavailable for routing."
            )


# Auto-register agents when module is imported
try:
    register_agents()
    logger.info("Agent registration complete")
except Exception as e:
    logger.error(f"Error during agent registration: {e}", exc_info=True)


# Export commonly used classes
__all__ = [
    "AgentRegistry",
    "URLExtractorAgent",
    "PDFProcessorAgent",
    "NotionAgent",
]