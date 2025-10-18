"""
BaseAgent - Abstract base class for all agents

This module provides the foundation for all specialized agents in the system.
It handles common functionality like prompt loading, validation, and provides
a consistent interface that all agents must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List
import logging
import json

import yaml
from jsonschema import validate, ValidationError

from src.core.config import get_config

logger = logging.getLogger(__name__)
settings = get_config()


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Provides:
    - Prompt loading from YAML files
    - Prompt validation against JSON schema
    - Common helper methods for agent capabilities
    - Abstract methods that subclasses must implement

    Subclasses must implement:
    - _build_system_prompt(): Build the system prompt from configuration
    - _register_tools(): Register tools available to this agent
    - process_message(): Process incoming messages

    Example usage:
        class URLExtractorAgent(BaseAgent):
            def __init__(self, prompt_file="specialized/url_extractor_agent.yaml"):
                super().__init__(prompt_file)
                # Additional initialization

            def _build_system_prompt(self) -> str:
                # Build prompt from self.prompt_config
                config = self.prompt_config["system_prompt"]
                return f"{config['role']}\\n\\nCapabilities: {config['capabilities']}"

            def _register_tools(self) -> List[Any]:
                return [TavilyExtractTool()]

            async def process_message(self, message, context):
                return {"status": "complete", "response": "Processed!"}
    """

    def __init__(self, prompt_file: str):
        """
        Initialize the agent by loading and validating its prompt configuration.

        Args:
            prompt_file: Path to the YAML prompt file relative to prompts/ directory.
                        Should include subdirectory (e.g., "specialized/agent.yaml").
                        Can be overridden for testing or variant configurations.

        Raises:
            FileNotFoundError: If prompt file doesn't exist
            ValidationError: If prompt doesn't match schema
            yaml.YAMLError: If YAML is malformed
        """
        self.prompt_file = prompt_file
        self.prompt_config = self._load_prompt(prompt_file)
        self._validate_prompt(self.prompt_config)

        # Extract common properties
        self.agent_name = self.prompt_config["agent"]["name"]
        self.agent_version = self.prompt_config["agent"]["version"]
        self.agent_description = self.prompt_config["agent"]["description"]

        # Build system prompt (implemented by subclass)
        self.system_prompt = self._build_system_prompt()

        # Register tools (implemented by subclass)
        self.tools = self._register_tools()

        logger.info(
            f"Initialized {self.agent_name} v{self.agent_version}"
        )

    def _load_prompt(self, prompt_file: str) -> Dict[str, Any]:
        """
        Load prompt configuration from YAML file.

        Args:
            prompt_file: Path relative to prompts/ directory
                        (e.g., "specialized/web_scraper.yaml")

        Returns:
            Dictionary containing the parsed YAML configuration

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        # Construct full path: prompts/ + subdirectory/filename
        prompt_path = Path("prompts") / prompt_file

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}\n"
                f"Expected location: {prompt_path.absolute()}"
            )

        # Load YAML
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            logger.debug(f"Loaded prompt from {prompt_path}")
            return config

        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {prompt_path}: {e}")
            raise

    def _validate_prompt(self, config: Dict[str, Any]) -> None:
        """
        Validate prompt configuration against JSON schema.

        Args:
            config: The loaded prompt configuration

        Raises:
            ValidationError: If configuration doesn't match schema
        """
        schema_path = Path("prompts/schemas/prompt_schema.json")

        if not schema_path.exists():
            logger.warning(
                f"Schema file not found: {schema_path}. Skipping validation."
            )
            return

        # Load schema
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)

        # Validate
        try:
            validate(instance=config, schema=schema)
            logger.debug(f"Prompt validation successful for {self.prompt_file}")

        except ValidationError as e:
            logger.error(
                f"Prompt validation failed for {self.prompt_file}: {e.message}"
            )
            raise

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt from the loaded configuration.

        This method should construct the complete system prompt that will be
        sent to the LLM, using the data from self.prompt_config.

        Returns:
            Complete system prompt as a string

        Example:
            def _build_system_prompt(self) -> str:
                config = self.prompt_config["system_prompt"]
                parts = [
                    config["role"],
                    "\\n\\nCapabilities:",
                    "\\n".join(f"- {cap}" for cap in config["capabilities"]),
                    "\\n\\nInstructions:",
                    config["instructions"]
                ]
                return "\\n".join(parts)
        """
        pass

    @abstractmethod
    def _register_tools(self) -> List[Any]:
        """
        Register and return the tools available to this agent.

        Returns:
            List of tool objects (LangChain Tool format or custom format)

        Example:
            def _register_tools(self) -> List[Any]:
                return [
                    TavilyExtractTool(api_key=settings.tavily_api_key),
                    NotionSaveTool(api_key=settings.notion_api_key)
                ]
        """
        pass

    @abstractmethod
    async def process_message(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an incoming message.

        Args:
            message: The user's message
            context: Conversation context including:
                - conversation_id: Unique conversation identifier
                - user_id: User identifier
                - message_metadata: Additional message metadata
                - conversation_state: Agent-specific state

        Returns:
            Dictionary containing:
            - status: "complete", "awaiting_input", "awaiting_approval", "error", "routing"
            - response: The agent's response message
            - should_stream: Whether to stream the response
            - metadata: Optional metadata about the response
            - next_agent: (if status="routing") Next agent to route to

        Example:
            async def process_message(self, message, context):
                # Detect URL
                url = extract_url(message)

                if not url:
                    return {
                        "status": "error",
                        "response": "No URL found in message",
                        "should_stream": False
                    }

                # Request approval
                if not context.get("approval_granted"):
                    return {
                        "status": "awaiting_approval",
                        "response": self.get_approval_message("extract_url", url=url),
                        "should_stream": False
                    }

                # Process
                result = await self.extract_content(url)

                return {
                    "status": "complete",
                    "response": f"Extracted {result['word_count']} words from {url}",
                    "should_stream": False,
                    "metadata": result
                }
        """
        pass

    # Helper methods

    def get_capabilities(self) -> List[str]:
        """
        Get list of agent capabilities.

        Returns:
            List of capability strings
        """
        return self.prompt_config["system_prompt"].get("capabilities", [])

    def get_constraints(self) -> List[str]:
        """
        Get list of agent constraints.

        Returns:
            List of constraint strings
        """
        return self.prompt_config["system_prompt"].get("constraints", [])

    def get_tools_config(self) -> List[Dict[str, Any]]:
        """
        Get tool configurations from prompt.

        Returns:
            List of tool configuration dictionaries
        """
        return self.prompt_config.get("tools", [])

    def requires_approval(self, tool_name: str) -> bool:
        """
        Check if a tool requires user approval.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool requires approval, False otherwise
        """
        tools_config = self.get_tools_config()
        for tool in tools_config:
            if tool["name"] == tool_name:
                return tool.get("requires_approval", False)

        return False

    def get_approval_message(self, tool_name: str, **kwargs) -> str:
        """
        Get formatted approval message for a tool.

        Args:
            tool_name: Name of the tool
            **kwargs: Values to format into the message template

        Returns:
            Formatted approval message

        Example:
            msg = self.get_approval_message("extract_url", url="https://example.com")
        """
        approval_messages = self.prompt_config.get("approval_messages", {})
        template = approval_messages.get(tool_name, "Do you approve this action?")

        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(
                f"Missing template variable in approval message for {tool_name}: {e}"
            )
            return template

    def get_error_message(self, error_type: str, **kwargs) -> str:
        """
        Get formatted error message.

        Args:
            error_type: Type of error
            **kwargs: Values to format into the message template

        Returns:
            Formatted error message

        Example:
            msg = self.get_error_message("extraction_failed", url="https://example.com")
        """
        error_messages = self.prompt_config.get("error_messages", {})
        template = error_messages.get(
            error_type,
            f"An error occurred: {error_type}"
        )

        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(
                f"Missing template variable in error message for {error_type}: {e}"
            )
            return template

    def get_routing_config(self) -> Dict[str, Any]:
        """
        Get routing configuration for agent handoffs.

        Returns:
            Routing configuration dictionary with:
            - fallback_to: Default agent to route to
            - handoff_rules: List of routing rules
        """
        return self.prompt_config.get("routing", {})

    def __repr__(self) -> str:
        """String representation of the agent"""
        return f"{self.__class__.__name__}(name='{self.agent_name}', version='{self.agent_version}')"
