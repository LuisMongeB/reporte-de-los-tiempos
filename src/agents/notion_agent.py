"""
NotionAgent - Specialized agent for saving content to Notion

This agent handles saving URLs, PDFs, and user notes to Notion.
It supports both quick bookmarks and detailed content saves with summaries.
"""

import logging
from typing import Any, Dict, List, Literal
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from operator import add

from src.agents.BaseAgent import BaseAgent
from src.core.config import get_config
from src.tools.notion_tools import NotionTools

logger = logging.getLogger(__name__)
settings = get_config()


class NotionAgentState(TypedDict):
    """State for the Notion agent subgraph."""
    # Core state (compatible with AgentState)
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_context: Dict[str, Any]
    current_response: str

    # Notion-specific fields
    workflow_step: str
    title: str
    content: str
    metadata: Dict[str, Any]
    content_source: str  # "routed" or "direct"
    content_type: str  # "URL bookmark", "PDF summary", "User Note", etc.


class NotionAgent(BaseAgent):
    """
    Specialized agent for saving content to Notion.

    Workflow:
    1. Receive content (from routing or direct user request)
    2. Prepare page title and content
    3. Request user approval
    4. Create Notion page
    5. Return confirmation with page URL

    Handles:
    - URLs with or without extracted content
    - PDFs with or without summaries
    - Direct user notes and text
    """

    def __init__(self, prompt_file: str = "specialized/notion_agent.yaml"):
        """
        Initialize the Notion agent.

        Args:
            prompt_file: Path to YAML prompt configuration
        """
        logger.info("Initializing NotionAgent...")

        # Initialize Notion tools BEFORE calling super().__init__()
        self.notion_tools = None
        if settings.notion_api_key and settings.notion_default_parent_page_id:
            try:
                self.notion_tools = NotionTools()
                logger.info("NotionTools initialized successfully")
            except ValueError as e:
                logger.warning(f"NotionTools not available: {e}")
        else:
            logger.warning("Notion not configured - API key or parent page ID missing")

        # Initialize BaseAgent
        super().__init__(prompt_file)

        # Build LangGraph subgraph
        self.subgraph = self._build_subgraph()

        logger.info("NotionAgent initialized successfully with LangGraph subgraph")

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt from the loaded YAML configuration.

        Returns:
            Complete system prompt string for the LLM
        """
        config = self.prompt_config["system_prompt"]

        parts = [
            config["role"],
            "",
            "**Capabilities:**"
        ]

        for cap in config.get("capabilities", []):
            parts.append(f"- {cap}")

        parts.append("")
        parts.append("**Constraints:**")

        for constraint in config.get("constraints", []):
            parts.append(f"- {constraint}")

        parts.append("")
        parts.append(f"**Tone:** {config.get('tone', 'professional and efficient')}")
        parts.append("")
        parts.append("**Instructions:**")
        parts.append(config.get("instructions", ""))

        return "\n".join(parts)

    def _generate_title_from_content(self, content: str) -> str:
        """
        Generate a title from content when user doesn't provide one.

        Extracts the first meaningful words or uses a timestamp-based title.

        Args:
            content: The content to generate a title from

        Returns:
            Generated title string
        """
        from datetime import datetime

        # Clean up the content
        clean_content = content.strip()

        # If content is a URL, use the domain
        if clean_content.startswith(("http://", "https://", "www.")):
            from urllib.parse import urlparse
            try:
                parsed = urlparse(clean_content if "://" in clean_content else f"http://{clean_content}")
                domain = parsed.netloc or parsed.path.split("/")[0]
                return f"Page from {domain}"
            except:
                pass

        # Otherwise, use first few words (up to 50 chars)
        if len(clean_content) > 50:
            title = clean_content[:47].rsplit(" ", 1)[0] + "..."
        else:
            title = clean_content if clean_content else f"Note from {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        return title

    def _register_tools(self) -> List[Any]:
        """
        Register tools available to the Notion agent.

        Returns:
            List of tool objects
        """
        tools = []

        if hasattr(self, 'notion_tools') and self.notion_tools:
            tools.append(self.notion_tools)

        return tools

    def _build_subgraph(self) -> StateGraph:
        """
        Build the LangGraph subgraph for Notion saving workflow.

        Workflow (fully autonomous):
        1. determine_source -> [handle_routed_content OR handle_direct_request]
        2. handle_routed_content -> create_page
        3. handle_direct_request -> create_page
        4. create_page -> END
        """
        workflow = StateGraph(NotionAgentState)

        # Add nodes
        workflow.add_node("determine_source", self._determine_source_node)
        workflow.add_node("handle_routed_content", self._handle_routed_content_node)
        workflow.add_node("handle_direct_request", self._handle_direct_request_node)
        workflow.add_node("create_page", self._create_page_node)

        # Set entry point
        workflow.set_entry_point("determine_source")

        # Conditional routing from determine_source
        workflow.add_conditional_edges(
            "determine_source",
            self._route_by_source,
            {
                "routed": "handle_routed_content",
                "direct": "handle_direct_request",
                "error": END
            }
        )

        # After handling routed content, go to create_page
        workflow.add_edge("handle_routed_content", "create_page")

        # After handling direct request, go directly to create_page (autonomous)
        workflow.add_edge("handle_direct_request", "create_page")

        # create_page ends the workflow
        workflow.add_edge("create_page", END)

        # Compile with checkpointing (no interrupts - fully autonomous)
        checkpointer = MemorySaver()
        graph = workflow.compile(
            checkpointer=checkpointer
        )

        logger.info("NotionAgent subgraph compiled successfully")
        return graph

    def _determine_source_node(self, state: NotionAgentState) -> Dict[str, Any]:
        """
        Node: Determine if content is routed or direct request.

        Returns updates to state.
        """
        logger.info("determine_source_node: Checking content source")

        # Check conversation context for routed metadata
        context = state.get("conversation_context", {})
        routed_metadata = context.get("routed_data", {})

        if routed_metadata and routed_metadata.get("action"):
            # This is routed content
            return {
                "content_source": "routed",
                "workflow_step": "handle_routed"
            }
        else:
            # This is direct user request
            return {
                "content_source": "direct",
                "workflow_step": "handle_direct"
            }

    def _route_by_source(self, state: NotionAgentState) -> Literal["routed", "direct", "error"]:
        """
        Conditional edge: Route based on content source.

        Returns next node name.
        """
        content_source = state.get("content_source", "")
        logger.info(f"Routing by source: {content_source}")

        if content_source == "routed":
            return "routed"
        elif content_source == "direct":
            return "direct"
        else:
            return "error"

    def _handle_routed_content_node(self, state: NotionAgentState) -> Dict[str, Any]:
        """
        Node: Handle content routed from URLExtractor or PDFProcessor.

        Returns updates to state.
        """
        logger.info("handle_routed_content_node: Processing routed content")

        # Get routed metadata from conversation context
        context = state.get("conversation_context", {})
        routed_metadata = context.get("routed_data", {})
        action = routed_metadata.get("action", "")

        # Handle URL content
        if action == "save_url":
            url = routed_metadata.get("url", "")
            title = routed_metadata.get("title", url)
            content_text = routed_metadata.get("content", "")

            if content_text:
                content = f"**URL:** {url}\n\n{content_text}"
                content_type = "URL with content"
            else:
                content = f"**URL:** {url}"
                content_type = "URL bookmark"

            metadata = {"url": url, "source": "URLExtractorAgent"}

        # Handle PDF content
        elif action == "save_pdf":
            file_name = routed_metadata.get("file_name", "document.pdf")
            title = file_name
            content_text = routed_metadata.get("content", "")
            extraction_method = routed_metadata.get("extraction_method", "")

            if content_text:
                content = f"**Filename:** {file_name}\n**Extraction Method:** {extraction_method}\n\n{content_text}"
                content_type = "PDF with summary"
            else:
                content = f"**Filename:** {file_name}"
                content_type = "PDF reference"

            metadata = {"filename": file_name, "source": "PDFProcessorAgent"}

        else:
            error_msg = self.get_error_message("invalid_content")
            return {
                "current_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "workflow_step": "error"
            }

        # Ready to create (no approval needed - autonomous)
        return {
            "title": title,
            "content": content,
            "content_type": content_type,
            "metadata": metadata,
            "current_response": "",  # Will be set by create_page_node
            "messages": [],
            "workflow_step": "ready_for_creation"
        }

    def _handle_direct_request_node(self, state: NotionAgentState) -> Dict[str, Any]:
        """
        Node: Handle direct user request to save content.

        Returns updates to state.
        """
        logger.info("handle_direct_request_node: Processing direct request")

        # Get the last user message
        messages = state.get("messages", [])
        if not messages:
            error_msg = "No message provided"
            return {
                "current_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "workflow_step": "error"
            }

        latest_message = messages[-1]
        message = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)

        # Parse title and content
        title = None
        content = message

        # Pattern 1: "Save this note to Notion: <content>"
        if ":" in message:
            parts = message.split(":", 1)
            if len(parts) == 2:
                potential_title = parts[0].strip()
                # Check if first part looks like an instruction
                if any(keyword in potential_title.lower() for keyword in ["save", "note", "notion", "create", "page"]):
                    content = parts[1].strip()
                    title = None
                else:
                    title = potential_title
                    content = parts[1].strip()

        # Pattern 2: "Create a page called <title> with <content>"
        if "called" in message.lower() and "with" in message.lower():
            try:
                called_idx = message.lower().index("called")
                with_idx = message.lower().index("with")
                if called_idx < with_idx:
                    title = message[called_idx + 6:with_idx].strip()
                    content = message[with_idx + 4:].strip()
            except (ValueError, IndexError):
                pass

        if not title:
            # Auto-generate a title from the content
            # Use first few words of content or a timestamp-based title
            title = self._generate_title_from_content(content)
            logger.info(f"Auto-generated title: {title}")

        # Have both title and content - ready to create
        return {
            "title": title,
            "content": content,
            "content_type": "User Note",
            "metadata": {},
            "current_response": "",  # Will be set by create_page_node
            "messages": [],
            "workflow_step": "ready_for_creation"
        }

    def _route_after_direct_request(self, state: NotionAgentState) -> Literal["needs_title", "ready"]:
        """
        Conditional edge: Route after processing direct request.

        Returns next node name.
        """
        workflow_step = state.get("workflow_step", "")
        logger.info(f"Routing after direct request: {workflow_step}")

        if workflow_step == "needs_title":
            return "needs_title"
        else:
            return "ready"

    def _await_title_node(self, state: NotionAgentState) -> Dict[str, Any]:
        """
        Node: Wait for user to provide title.

        Returns updates to state.
        """
        logger.info("await_title_node: Processing user-provided title")

        # Get the last user message
        messages = state.get("messages", [])
        if messages:
            latest_message = messages[-1]
            title = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
            title = title.strip()

            if not title:
                error_msg = "Please provide a title for the Notion page."
                return {
                    "current_response": error_msg,
                    "messages": [AIMessage(content=error_msg)],
                    "workflow_step": "needs_title"
                }

            # Got title, prepare for creation
            content = state.get("content", "")
            approval_msg = self.get_approval_message(
                "create_notion_page",
                title=title,
                content_type="User Note"
            )

            return {
                "title": title,
                "content_type": "User Note",
                "metadata": {},
                "current_response": approval_msg,
                "messages": [AIMessage(content=approval_msg)],
                "workflow_step": "ready_for_creation"
            }

        error_msg = "No title provided"
        return {
            "current_response": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "workflow_step": "error"
        }

    async def _create_page_node(self, state: NotionAgentState) -> Dict[str, Any]:
        """
        Node: Create Notion page.

        Returns updates to state.
        """
        logger.info("create_page_node: Creating Notion page")

        # Check if Notion is configured
        if not self.notion_tools:
            error_msg = self.get_error_message("notion_not_configured")
            return {
                "current_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "workflow_step": "error"
            }

        title = state.get("title", "Untitled")
        content = state.get("content", "")
        metadata = state.get("metadata", {})

        try:
            # Create the page
            result = await self.notion_tools.create_page(
                title=title,
                content=content,
                metadata=metadata
            )

            if not result['success']:
                error_msg = self.get_error_message(
                    "creation_failed",
                    error=result.get('error', 'Unknown error')
                )
                return {
                    "current_response": error_msg,
                    "messages": [AIMessage(content=error_msg)],
                    "workflow_step": "error"
                }

            # Success
            page_url = result['page_url']
            confirmation = f"""✅ **Saved to Notion**

                **Page:** {title}
                **URL:** {page_url}

                Your content has been saved successfully!"""

            return {
                "current_response": confirmation,
                "messages": [AIMessage(content=confirmation)],
                "workflow_step": "complete"
            }

        except Exception as e:
            logger.error(f"Error creating Notion page: {e}", exc_info=True)
            error_msg = self.get_error_message("creation_failed", error=str(e))
            return {
                "current_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "workflow_step": "error"
            }

    async def process_message(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an incoming message for saving to Notion.

        Handles multiple scenarios:
        1. Routed from URLExtractorAgent with URL data
        2. Routed from PDFProcessorAgent with PDF data
        3. Direct user request to save content

        Args:
            message: The user's message
            context: Conversation context including:
                - conversation_id: Unique conversation identifier
                - user_id: User identifier
                - conversation_state: Agent-specific state
                - approval_granted: Whether user approved the action
                - metadata: Content from routing agents (URL, PDF info, etc.)

        Returns:
            Dictionary containing:
            - status: "complete", "awaiting_input", "awaiting_approval", "error"
            - response: The agent's response message
            - should_stream: Whether to stream the response
            - metadata: Optional metadata
            - conversation_state: Updated state for next interaction
        """
        logger.info(f"NotionAgent processing message: {message[:100]}...")

        # Check if Notion is configured
        if not self.notion_tools:
            error_msg = self.get_error_message("notion_not_configured")
            return {
                "status": "error",
                "response": error_msg,
                "should_stream": False,
                "conversation_state": {}
            }

        # Get current conversation state
        state = context.get("conversation_state", {})
        workflow_step = state.get("step", "determine_source")

        # Step 1: Determine content source and prepare page
        if workflow_step == "determine_source":
            # Check if this is routed content (from URL or PDF agents)
            routed_metadata = context.get("metadata", {})

            if routed_metadata:
                # This is routed content from another agent
                return await self._handle_routed_content(routed_metadata, context)
            else:
                # This is a direct user request
                return await self._handle_direct_request(message, context)

        # Step 2: Create page after approval
        if workflow_step == "create_page_approved":
            if not context.get("approval_granted"):
                return {
                    "status": "complete",
                    "response": "Okay, I won't save to Notion. Let me know if you need anything else!",
                    "should_stream": False,
                    "conversation_state": {}
                }

            # Get prepared content from state
            title = state.get("title")
            content = state.get("content")
            metadata = state.get("metadata", {})

            # Create the Notion page
            result = await self._create_and_confirm(title, content, metadata)
            return result

        # Step 3: User needs to provide title for direct request
        if workflow_step == "awaiting_title":
            title = message.strip()

            if not title:
                return {
                    "status": "awaiting_input",
                    "response": "Please provide a title for the Notion page.",
                    "should_stream": False,
                    "conversation_state": state
                }

            # Now we have title, prepare for approval
            content = state.get("content", "")
            approval_msg = self.get_approval_message(
                "create_notion_page",
                title=title,
                content_type="User Note"
            )

            return {
                "status": "awaiting_approval",
                "response": approval_msg,
                "should_stream": False,
                "conversation_state": {
                    "step": "create_page_approved",
                    "title": title,
                    "content": content,
                    "metadata": {}
                }
            }

        # Unknown state - reset
        return {
            "status": "error",
            "response": "I'm sorry, I lost track of our conversation. Please try again.",
            "should_stream": False,
            "conversation_state": {}
        }

    async def _handle_routed_content(
        self,
        routed_metadata: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle content routed from URLExtractor or PDFProcessor agents.

        Args:
            routed_metadata: Metadata from routing agent
            context: Conversation context

        Returns:
            Response dictionary with approval request
        """
        action = routed_metadata.get("action", "")

        # Handle URL content
        if action == "save_url":
            url = routed_metadata.get("url", "")
            title = routed_metadata.get("title", url)
            content_text = routed_metadata.get("content", "")

            # Prepare page content
            if content_text:
                # Full content save
                content = f"**URL:** {url}\n\n{content_text}"
                content_type = "URL with content"
            else:
                # Quick bookmark
                content = f"**URL:** {url}"
                content_type = "URL bookmark"

            metadata = {"url": url, "source": "URLExtractorAgent"}

        # Handle PDF content
        elif action == "save_pdf":
            file_name = routed_metadata.get("file_name", "document.pdf")
            title = file_name
            content_text = routed_metadata.get("content", "")
            extraction_method = routed_metadata.get("extraction_method", "")

            # Prepare page content
            if content_text:
                # Full content save
                content = f"**Filename:** {file_name}\n**Extraction Method:** {extraction_method}\n\n{content_text}"
                content_type = "PDF with summary"
            else:
                # Just the reference
                content = f"**Filename:** {file_name}"
                content_type = "PDF reference"

            metadata = {"filename": file_name, "source": "PDFProcessorAgent"}

        else:
            return {
                "status": "error",
                "response": self.get_error_message("invalid_content"),
                "should_stream": False,
                "conversation_state": {}
            }

        # Request approval
        approval_msg = self.get_approval_message(
            "create_notion_page",
            title=title,
            content_type=content_type
        )

        return {
            "status": "awaiting_approval",
            "response": approval_msg,
            "should_stream": False,
            "conversation_state": {
                "step": "create_page_approved",
                "title": title,
                "content": content,
                "metadata": metadata
            }
        }

    async def _handle_direct_request(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle direct user request to save content.

        Parses messages like:
        - "Save this note to Notion: Meeting notes from today"
        - "Create a page called Project Ideas with ..."

        Args:
            message: User's message
            context: Conversation context

        Returns:
            Response dictionary
        """
        # Try to parse title and content from message
        title = None
        content = message

        # Pattern 1: "Save this note to Notion: <content>"
        if ":" in message:
            parts = message.split(":", 1)
            if len(parts) == 2:
                potential_title = parts[0].strip()
                # Check if first part looks like an instruction
                if any(keyword in potential_title.lower() for keyword in ["save", "note", "notion", "create", "page"]):
                    content = parts[1].strip()
                    title = None  # Ask user for title
                else:
                    # First part might be the title
                    title = potential_title
                    content = parts[1].strip()

        # Pattern 2: "Create a page called <title> with <content>"
        if "called" in message.lower() and "with" in message.lower():
            try:
                called_idx = message.lower().index("called")
                with_idx = message.lower().index("with")
                if called_idx < with_idx:
                    title = message[called_idx + 6:with_idx].strip()
                    content = message[with_idx + 4:].strip()
            except (ValueError, IndexError):
                pass

        if not title:
            # Ask user for title
            return {
                "status": "awaiting_input",
                "response": "What should I use as the page title?",
                "should_stream": False,
                "conversation_state": {
                    "step": "awaiting_title",
                    "content": content
                }
            }

        # We have both title and content, request approval
        approval_msg = self.get_approval_message(
            "create_notion_page",
            title=title,
            content_type="User Note"
        )

        return {
            "status": "awaiting_approval",
            "response": approval_msg,
            "should_stream": False,
            "conversation_state": {
                "step": "create_page_approved",
                "title": title,
                "content": content,
                "metadata": {}
            }
        }

    async def _create_and_confirm(
        self,
        title: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create Notion page and return confirmation.

        Args:
            title: Page title
            content: Page content
            metadata: Page metadata

        Returns:
            Response dictionary with confirmation or error
        """
        logger.info(f"Creating Notion page: {title}")

        try:
            # Create the page
            result = await self.notion_tools.create_page(
                title=title,
                content=content,
                metadata=metadata
            )

            if not result['success']:
                error_msg = self.get_error_message(
                    "creation_failed",
                    error=result.get('error', 'Unknown error')
                )
                return {
                    "status": "error",
                    "response": error_msg,
                    "should_stream": False,
                    "conversation_state": {}
                }

            # Success - return confirmation
            page_url = result['page_url']
            confirmation = f"""✅ **Saved to Notion**

**Page:** {title}
**URL:** {page_url}

Your content has been saved successfully!"""

            return {
                "status": "complete",
                "response": confirmation,
                "should_stream": False,
                "metadata": {
                    "page_id": result['page_id'],
                    "page_url": page_url,
                    "title": title
                },
                "conversation_state": {}
            }

        except Exception as e:
            logger.error(f"Error creating Notion page: {e}", exc_info=True)
            error_msg = self.get_error_message("creation_failed", error=str(e))
            return {
                "status": "error",
                "response": error_msg,
                "should_stream": False,
                "conversation_state": {}
            }
