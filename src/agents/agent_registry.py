"""
Agent Registry System

This module provides a centralized registry for all agents in the system,
enabling dynamic agent discovery, instantiation, and capability-based routing.

Key Features:
- Dynamic agent registration via decorator pattern
- Lazy agent instantiation with singleton caching
- Capability-based agent discovery
- Metadata access without instantiation
- Health checks based on required dependencies
- Support for multiple agents with different configurations

Usage:
    # Register an agent
    @register_agent(
        agent_id="url_extractor",
        agent_class=URLExtractorAgent,
        prompt_file="prompts/specialized/url_extractor_agent.yaml",
        required_settings=["tavily_api_key"],
        categories=["content_extraction"]
    )

    # Get registry instance
    registry = AgentRegistry.get_instance()

    # Discover agents
    agent = registry.get_agent("url_extractor")
    agents = registry.find_by_capability("Extract content from URLs")
    available = registry.get_available_agents()

    # Query metadata
    info = registry.get_agent_info("url_extractor")
    is_available = registry.is_agent_available("url_extractor")
"""

import logging
from typing import Dict, List, Optional, Type, Any, Callable
from pathlib import Path
import yaml
from functools import lru_cache

from src.agents.BaseAgent import BaseAgent
from src.core.config import get_config

logger = logging.getLogger(__name__)


class AgentRegistryError(Exception):
    """Base exception for agent registry errors."""
    pass


class AgentNotFoundError(AgentRegistryError):
    """Raised when an agent is not found in the registry."""
    pass


class AgentNotAvailableError(AgentRegistryError):
    """Raised when an agent cannot be instantiated due to missing dependencies."""
    pass


class AgentMetadata:
    """
    Metadata for a registered agent.

    This class holds configuration and metadata about an agent without
    instantiating the agent itself, enabling efficient agent discovery.

    Attributes:
        agent_id: Unique identifier for the agent
        agent_class: The agent class (not instantiated)
        prompt_file: Path to YAML config relative to prompts/ (e.g., "shared/test_agent.yaml")
        required_settings: List of required settings keys from config
        categories: List of categories/tags for the agent
        prompt_config: Loaded YAML configuration (cached)
    """

    def __init__(
        self,
        agent_id: str,
        agent_class: Type[BaseAgent],
        prompt_file: str,
        required_settings: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ):
        """
        Initialize agent metadata.

        Args:
            agent_id: Unique identifier for the agent
            agent_class: The agent class to instantiate
            prompt_file: Path relative to prompts/ directory (e.g., "specialized/url_extractor_agent.yaml")
            required_settings: List of required config settings (e.g., ["tavily_api_key"])
            categories: List of categories (e.g., ["content_extraction", "web"])
        """
        self.agent_id = agent_id
        self.agent_class = agent_class
        self.prompt_file = prompt_file
        self.required_settings = required_settings or []
        self.categories = categories or []
        self._prompt_config: Optional[Dict[str, Any]] = None

    @property
    def prompt_config(self) -> Dict[str, Any]:
        """
        Load and cache the prompt configuration.

        Returns:
            Dict containing the parsed YAML configuration

        Raises:
            AgentRegistryError: If the prompt file cannot be loaded
        """
        if self._prompt_config is None:
            try:
                # Construct full path: prompts/ + subdirectory/filename
                # (matching BaseAgent behavior)
                prompt_path = Path("prompts") / self.prompt_file
                if not prompt_path.exists():
                    raise AgentRegistryError(f"Prompt file not found: {prompt_path}")

                with open(prompt_path, 'r') as f:
                    self._prompt_config = yaml.safe_load(f)

            except Exception as e:
                raise AgentRegistryError(f"Failed to load prompt file {self.prompt_file}: {e}")

        return self._prompt_config

    @property
    def name(self) -> str:
        """Get the agent's display name from configuration."""
        return self.prompt_config.get("agent", {}).get("name", self.agent_id)

    @property
    def version(self) -> str:
        """Get the agent's version from configuration."""
        return self.prompt_config.get("agent", {}).get("version", "unknown")

    @property
    def description(self) -> str:
        """Get the agent's description from configuration."""
        return self.prompt_config.get("agent", {}).get("description", "")

    @property
    def capabilities(self) -> List[str]:
        """Get the agent's capabilities from configuration."""
        return self.prompt_config.get("system_prompt", {}).get("capabilities", [])

    @property
    def constraints(self) -> List[str]:
        """Get the agent's constraints from configuration."""
        return self.prompt_config.get("system_prompt", {}).get("constraints", [])

    def is_available(self) -> bool:
        """
        Check if the agent can be instantiated.

        Returns:
            True if all required settings are available, False otherwise
        """
        config = get_config()

        for setting_key in self.required_settings:
            if not hasattr(config, setting_key):
                logger.debug(f"Agent {self.agent_id}: missing setting {setting_key}")
                return False

            setting_value = getattr(config, setting_key)
            if setting_value is None or setting_value == "":
                logger.debug(f"Agent {self.agent_id}: setting {setting_key} is empty")
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to a dictionary.

        Returns:
            Dict with agent metadata
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": self.capabilities,
            "constraints": self.constraints,
            "categories": self.categories,
            "required_settings": self.required_settings,
            "is_available": self.is_available()
        }


class AgentRegistry:
    """
    Centralized registry for all agents in the system.

    This class manages agent registration, discovery, and instantiation.
    It uses a singleton pattern to ensure a single registry instance
    across the application.

    Features:
    - Dynamic agent registration
    - Lazy instantiation with caching
    - Capability-based discovery
    - Health checks
    - Metadata queries without instantiation

    Example:
        registry = AgentRegistry.get_instance()

        # Register an agent
        registry.register(
            agent_id="url_extractor",
            agent_class=URLExtractorAgent,
            prompt_file="prompts/specialized/url_extractor_agent.yaml",
            required_settings=["tavily_api_key"]
        )

        # Get an agent instance
        agent = registry.get_agent("url_extractor")

        # Discover agents by capability
        agents = registry.find_by_capability("Extract content from URLs")
    """

    _instance: Optional['AgentRegistry'] = None

    def __init__(self):
        """
        Initialize the agent registry.

        Note: Use get_instance() instead of direct instantiation.
        """
        self._metadata: Dict[str, AgentMetadata] = {}
        self._instances: Dict[str, BaseAgent] = {}
        logger.info("AgentRegistry initialized")

    @classmethod
    def get_instance(cls) -> 'AgentRegistry':
        """
        Get the singleton registry instance.

        Returns:
            The singleton AgentRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.

        Useful for testing to ensure a clean state.
        """
        cls._instance = None

    def register(
        self,
        agent_id: str,
        agent_class: Type[BaseAgent],
        prompt_file: str,
        required_settings: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> None:
        """
        Register an agent with the registry.

        Args:
            agent_id: Unique identifier for the agent
            agent_class: The agent class to instantiate
            prompt_file: Path to the YAML prompt configuration
            required_settings: List of required config settings
            categories: List of categories/tags

        Raises:
            AgentRegistryError: If agent_id is already registered
        """
        if agent_id in self._metadata:
            logger.warning(f"Agent {agent_id} is already registered, overwriting")

        metadata = AgentMetadata(
            agent_id=agent_id,
            agent_class=agent_class,
            prompt_file=prompt_file,
            required_settings=required_settings,
            categories=categories
        )

        self._metadata[agent_id] = metadata
        logger.info(f"Registered agent: {agent_id} ({metadata.name} v{metadata.version})")

    def unregister(self, agent_id: str) -> None:
        """
        Unregister an agent from the registry.

        Args:
            agent_id: The agent ID to unregister

        Raises:
            AgentNotFoundError: If agent is not registered
        """
        if agent_id not in self._metadata:
            raise AgentNotFoundError(f"Agent not found: {agent_id}")

        # Remove metadata
        del self._metadata[agent_id]

        # Remove cached instance if exists
        if agent_id in self._instances:
            del self._instances[agent_id]

        logger.info(f"Unregistered agent: {agent_id}")

    def get_agent(self, agent_id: str, force_new: bool = False) -> BaseAgent:
        """
        Get an agent instance.

        Agents are instantiated lazily and cached. By default, returns
        the cached instance if available.

        Args:
            agent_id: The agent ID to get
            force_new: If True, creates a new instance even if cached

        Returns:
            The agent instance

        Raises:
            AgentNotFoundError: If agent is not registered
            AgentNotAvailableError: If agent cannot be instantiated
        """
        if agent_id not in self._metadata:
            raise AgentNotFoundError(f"Agent not found: {agent_id}")

        metadata = self._metadata[agent_id]

        # Check if agent is available
        if not metadata.is_available():
            config = get_config()
            missing = [s for s in metadata.required_settings
                      if not hasattr(config, s) or getattr(config, s) is None]
            raise AgentNotAvailableError(
                f"Agent {agent_id} is not available. Missing settings: {missing}"
            )

        # Return cached instance if available and not forcing new
        if not force_new and agent_id in self._instances:
            logger.debug(f"Returning cached instance of {agent_id}")
            return self._instances[agent_id]

        # Instantiate the agent
        try:
            logger.info(f"Instantiating agent: {agent_id}")
            agent = metadata.agent_class(prompt_file=metadata.prompt_file)

            # Cache the instance
            self._instances[agent_id] = agent

            return agent

        except Exception as e:
            raise AgentRegistryError(f"Failed to instantiate agent {agent_id}: {e}")

    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent metadata without instantiating.

        Args:
            agent_id: The agent ID to query

        Returns:
            Dict with agent metadata

        Raises:
            AgentNotFoundError: If agent is not registered
        """
        if agent_id not in self._metadata:
            raise AgentNotFoundError(f"Agent not found: {agent_id}")

        return self._metadata[agent_id].to_dict()

    def is_agent_available(self, agent_id: str) -> bool:
        """
        Check if an agent is available for use.

        Args:
            agent_id: The agent ID to check

        Returns:
            True if agent can be instantiated, False otherwise
        """
        if agent_id not in self._metadata:
            return False

        return self._metadata[agent_id].is_available()

    def list_agents(self, include_unavailable: bool = True) -> List[str]:
        """
        List all registered agent IDs.

        Args:
            include_unavailable: If False, only returns available agents

        Returns:
            List of agent IDs
        """
        if include_unavailable:
            return list(self._metadata.keys())

        return [
            agent_id for agent_id, metadata in self._metadata.items()
            if metadata.is_available()
        ]

    def get_available_agents(self) -> List[Dict[str, Any]]:
        """
        Get information about all available agents.

        Returns:
            List of dicts with agent metadata for available agents
        """
        return [
            metadata.to_dict()
            for metadata in self._metadata.values()
            if metadata.is_available()
        ]

    def find_by_capability(self, capability_pattern: str) -> List[str]:
        """
        Find agents by capability.

        Searches agent capabilities for a substring match (case-insensitive).

        Args:
            capability_pattern: The capability string to search for

        Returns:
            List of agent IDs that have matching capabilities
        """
        pattern_lower = capability_pattern.lower()
        matching_agents = []

        for agent_id, metadata in self._metadata.items():
            if not metadata.is_available():
                continue

            for capability in metadata.capabilities:
                if pattern_lower in capability.lower():
                    matching_agents.append(agent_id)
                    break

        return matching_agents

    def find_by_category(self, category: str) -> List[str]:
        """
        Find agents by category.

        Args:
            category: The category to search for

        Returns:
            List of agent IDs in that category
        """
        return [
            agent_id for agent_id, metadata in self._metadata.items()
            if category in metadata.categories and metadata.is_available()
        ]

    def clear_cache(self, agent_id: Optional[str] = None) -> None:
        """
        Clear cached agent instances.

        Args:
            agent_id: If provided, only clears that agent's cache.
                     If None, clears all cached instances.
        """
        if agent_id:
            if agent_id in self._instances:
                del self._instances[agent_id]
                logger.info(f"Cleared cache for agent: {agent_id}")
        else:
            self._instances.clear()
            logger.info("Cleared all agent caches")


# Decorator for registering agents
def register_agent(
    agent_id: str,
    agent_class: Type[BaseAgent],
    prompt_file: str,
    required_settings: Optional[List[str]] = None,
    categories: Optional[List[str]] = None
) -> Callable:
    """
    Decorator for registering an agent with the registry.

    This decorator automatically registers an agent class when the module
    is imported, making agent registration declarative and easy to manage.

    Args:
        agent_id: Unique identifier for the agent
        agent_class: The agent class to register
        prompt_file: Path to the YAML prompt configuration
        required_settings: List of required config settings
        categories: List of categories/tags

    Returns:
        The original class (unmodified)

    Example:
        @register_agent(
            agent_id="url_extractor",
            agent_class=URLExtractorAgent,
            prompt_file="prompts/specialized/url_extractor_agent.yaml",
            required_settings=["tavily_api_key"],
            categories=["content_extraction", "web"]
        )
        class URLExtractorAgent(BaseAgent):
            pass
    """
    def decorator(cls):
        registry = AgentRegistry.get_instance()
        registry.register(
            agent_id=agent_id,
            agent_class=agent_class,
            prompt_file=prompt_file,
            required_settings=required_settings,
            categories=categories
        )
        return cls

    return decorator
