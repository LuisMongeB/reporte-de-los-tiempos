"""
Supervisor agent for processing user messages and generating responses.
"""

import logging
import re
from operator import add
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from src.core.config import get_config
from src.agents.BaseAgent import BaseAgent
from src.agents.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)
settings = get_config()


class RoutingDecision(BaseModel):
    """
    Structured output from LLM for intelligent routing decisions.

    This model uses Pydantic to ensure type-safe, deterministic routing
    decisions from the LLM. The LLM analyzes user messages and returns
    this structured format instead of free text.

    This is the 2025 production best practice for LLM-based agent routing.
    """

    intent: Literal["url", "pdf", "notion", "general"] = Field(
        description=(
            "The PRIMARY intent of the user's message:\n"
            "- 'url': Extract/analyze content from a web URL\n"
            "- 'pdf': Process or analyze a PDF document\n"
            "- 'notion': Save/upload/store content to Notion\n"
            "- 'general': General conversation or questions"
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0) in the routing decision"
    )

    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )

    extracted_url: Optional[str] = Field(
        default=None,
        description="If a URL is detected in the message, extract it here (with or without http://)"
    )

    notion_action: Optional[str] = Field(
        default=None,
        description="If intent is 'notion', what action? (save, upload, create, etc.)"
    )


class AgentState(TypedDict):
    """
    State maintained throughout the agent's execution.

    LangGraph passes this state between nodes. Each node can read and modify it.
    This state supports multi-agent routing and orchestration.

    Reducers:
    - messages: Uses add_messages to append new messages to history (never overwrites)
    - routed_data: Merges dicts (new values update existing keys)
    - routing_history: Appends new routing events to history list
    """

    # Core conversation state
    messages: Annotated[List[BaseMessage], add_messages]  # ✅ Reducer appends messages
    conversation_context: Dict[str, Any]  # Additional context (user info, etc.)
    current_response: str  # The agent's current response being built

    # Routing state (for multi-agent orchestration)
    current_agent: str  # Which agent is currently handling the message (default: "supervisor")
    routed_data: Annotated[Dict[str, Any], lambda x, y: {**x, **y}]  # ✅ Reducer merges dicts
    routing_history: Annotated[List[Dict[str, str]], add]  # ✅ Reducer appends to list


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that processes user messages and generates responses.

    For MVP, this agent handles all messages directly. In future phases,
    it will route to specialized agents based on message content.

    Extends BaseAgent to use YAML-based prompts and consistent interface.
    """

    def __init__(self, prompt_file: str = "supervisor/supervisor_agent.yaml"):
        """
        Initialize the supervisor agent with LLM and graph.

        Args:
            prompt_file: Path to YAML prompt configuration (default: supervisor/supervisor_agent.yaml)
        """
        logger.info("Initializing SupervisorAgent...")

        # Initialize BaseAgent (loads prompt, validates, builds system prompt)
        super().__init__(prompt_file)

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Build the agent graph
        self.graph = self._build_graph()

        logger.info("SupervisorAgent initialized successfully")

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt from the loaded YAML configuration.

        Returns:
            Complete system prompt string for the LLM
        """
        config = self.prompt_config["system_prompt"]

        # Build comprehensive system prompt
        parts = [
            config["role"],
            "",
            "**Capabilities:**"
        ]

        # Add capabilities
        for cap in config.get("capabilities", []):
            parts.append(f"- {cap}")

        parts.append("")
        parts.append("**Constraints:**")

        # Add constraints
        for constraint in config.get("constraints", []):
            parts.append(f"- {constraint}")

        parts.append("")
        parts.append(f"**Tone:** {config.get('tone', 'professional and helpful')}")
        parts.append("")
        parts.append("**Instructions:**")
        parts.append(config.get("instructions", ""))

        return "\n".join(parts)

    def _register_tools(self) -> List[Any]:
        """
        Register tools available to the supervisor agent.

        For MVP, the supervisor doesn't use external tools yet.
        In Phase 2, this will include routing tools.

        Returns:
            List of tool objects (empty for MVP)
        """
        # No tools in MVP - supervisor uses LLM directly
        # Phase 2 will add: route_to_agent, request_approval
        return []

    async def process_message(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an incoming message (BaseAgent interface).

        This is the new interface required by BaseAgent. It wraps the
        existing process() method for backward compatibility.

        Args:
            message: The user's message
            context: Conversation context

        Returns:
            Dictionary with status, response, should_stream, etc.
        """
        # Extract conversation history and user context
        conversation_history = context.get("conversation_history", [])
        user_context = context.get("user_context", {})

        # Use existing process() method
        response = await self.process(message, conversation_history, user_context)

        return {
            "status": "complete",
            "response": response,
            "should_stream": True,  # Enable streaming for better UX
            "metadata": {}
        }

    def _initialize_llm(self):
        """
        Initialize the OpenAI LLM.

        Returns:
            ChatOpenAI: Configured LLM instance
        """
        logger.info(f"Initializing OpenAI model: {settings.llm_model}")
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            max_tokens=settings.llm_max_tokens,
        )

    async def _detect_message_type(self, message: str) -> str:
        """
        Use LLM intelligence to detect the message intent and determine routing.

        This method uses Pydantic structured outputs with the LLM to intelligently
        analyze user messages and determine routing. This is the 2025 production
        best practice for agent routing, replacing brittle keyword/regex patterns.

        The LLM analyzes the message to identify:
        - URL extraction/processing requests (for URLExtractorAgent)
        - PDF processing requests (for PDFProcessorAgent)
        - Notion saving/storage requests (for NotionAgent)
        - General conversation (handled by supervisor)

        Key advantages over pattern matching:
        - Understands natural language variations ("upload", "save", "add", etc.)
        - Context-aware (knows "upload URL to Notion" routes to NotionAgent)
        - Can detect URLs without http:// prefix (www.example.com)
        - Type-safe with Pydantic validation

        Args:
            message: The user's message

        Returns:
            Message type: "url", "pdf", "notion", or "general"
        """
        try:
            # Create structured output LLM for routing decisions
            router_llm = self.llm.with_structured_output(RoutingDecision)

            # Build routing prompt
            routing_prompt = f"""Analyze the following user message and determine its PRIMARY intent for routing to the appropriate agent.

            **Available Agents:**
            - **URL Agent**: Extracts and analyzes content from web URLs/links
            - **PDF Agent**: Processes and analyzes PDF documents
            - **Notion Agent**: Saves/uploads/stores content to Notion
            - **Supervisor (General)**: Handles general conversation and questions

            **Important Routing Rules:**
            1. If the message mentions saving/uploading/storing something TO Notion → choose "notion"
            2. If the message contains a URL but ALSO mentions Notion → choose "notion" (destination is primary)
            3. If the message only contains a URL without storage intent → choose "url"
            4. If the message mentions PDF processing → choose "pdf"
            5. General questions or conversation → choose "general"

            **User Message:**
            "{message}"

            Analyze this message carefully and return a structured routing decision."""

            # Get structured routing decision from LLM
            decision: RoutingDecision = await router_llm.ainvoke([
                SystemMessage(content="You are an intelligent routing agent. Analyze user messages and determine the correct agent to handle them."),
                HumanMessage(content=routing_prompt)
            ])

            # Log the decision
            logger.info(
                f"LLM Routing Decision: intent='{decision.intent}', "
                f"confidence={decision.confidence:.2f}, "
                f"reasoning='{decision.reasoning[:100]}...'"
            )

            # Log extracted metadata
            if decision.extracted_url:
                logger.debug(f"Extracted URL: {decision.extracted_url}")
            if decision.notion_action:
                logger.debug(f"Notion action: {decision.notion_action}")

            return decision.intent

        except Exception as e:
            logger.error(f"Error in LLM-based routing: {e}", exc_info=True)
            logger.warning("Falling back to pattern-based routing")
            # Fallback to simple pattern matching
            return self._detect_message_type_fallback(message)

    def _detect_message_type_fallback(self, message: str) -> str:
        """
        Fallback pattern-based detection when LLM routing fails.

        This is a safety net for when the LLM is unavailable or returns errors.
        Uses simple regex/keyword matching as a last resort.

        Args:
            message: The user's message

        Returns:
            Message type: "url", "pdf", "notion", or "general"
        """
        routing_config = self.prompt_config.get("routing", {})
        patterns = routing_config.get("detection_patterns", {})

        # Check URL pattern (search anywhere in message, not just at start)
        url_pattern = patterns.get("url_pattern", r"https?://\S+")
        if re.search(url_pattern, message):
            logger.info(f"Fallback: Detected URL message: {message[:50]}...")
            return "url"

        # Check PDF pattern
        pdf_pattern = patterns.get("pdf_pattern", r".*\.pdf$")
        if re.match(pdf_pattern, message.strip(), re.IGNORECASE):
            logger.info(f"Fallback: Detected PDF message: {message[:50]}...")
            return "pdf"

        # Check Notion keywords
        notion_keywords = patterns.get("notion_keywords", [])
        message_lower = message.lower()
        if any(keyword.lower() in message_lower for keyword in notion_keywords):
            logger.info(f"Fallback: Detected Notion keyword in message: {message[:50]}...")
            return "notion"

        logger.debug(f"Fallback: Message classified as general: {message[:50]}...")
        return "general"

    def _determine_routing(self, message_type: str, message: str) -> Optional[str]:
        """
        Determine which agent to route to, considering agent availability.

        Implements smart fallback logic:
        - URLs: Try URLExtractor first, fall back to NotionAgent for bookmarking
        - PDFs: Route to PDFProcessor if available
        - Notion: Route to NotionAgent if available
        - General: Handle in supervisor (returns None)

        Args:
            message_type: The detected message type
            message: The original message (for logging)

        Returns:
            agent_id to route to, or None if should handle in supervisor
        """
        registry = AgentRegistry.get_instance()
        agent_map = self.prompt_config.get("routing", {}).get("agent_mapping", {})

        # Special case for URLs: can fall back to Notion for bookmarking
        if message_type == "url":
            if registry.is_agent_available("url_extractor"):
                logger.info("Routing URL to URLExtractorAgent")
                return "url_extractor"
            elif registry.is_agent_available("notion_agent"):
                logger.info("URLExtractor unavailable, routing URL to NotionAgent for bookmarking")
                return "notion_agent"
            else:
                logger.warning("Both URLExtractor and NotionAgent unavailable for URL")
                return None

        # For other types, just check primary agent
        primary_agent = agent_map.get(message_type)
        if primary_agent and registry.is_agent_available(primary_agent):
            logger.info(f"Routing {message_type} message to {primary_agent}")
            return primary_agent

        if primary_agent:
            logger.warning(f"Agent {primary_agent} is not available for {message_type} message")

        return None


    async def _detect_and_route(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph node: Detect message type and route to appropriate agent.

        This is the routing node in the LangGraph workflow. It:
        1. Extracts the latest user message
        2. Detects message type
        3. Determines which agent should handle it
        4. Routes to specialized agent OR continues to supervisor processing

        Args:
            state: Current agent state

        Returns:
            Dict with state updates (current_agent, and optionally error messages)
        """
        logger.info("=== ENTERED _detect_and_route node ===")
        logger.debug(f"State keys: {state.keys()}")
        logger.debug(f"Number of messages: {len(state.get('messages', []))}")

        # Extract latest message (last message in the list)
        if not state["messages"]:
            logger.warning("No messages in state for routing detection")
            return {"current_agent": "supervisor"}

        latest_message = state["messages"][-1]
        message_content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)

        logger.info(f"Latest message type: {type(latest_message)}")
        logger.info(f"Message content to route: '{message_content[:100]}'")

        # Detect message type using LLM intelligence
        logger.info("About to call _detect_message_type...")
        message_type = await self._detect_message_type(message_content)
        logger.info(f"Detected message type: {message_type}")

        # Determine routing
        agent_id = self._determine_routing(message_type, message_content)

        if agent_id:
            # Mark for routing to specialized agent
            logger.info(f"Routing decision: {agent_id}")
            return {"current_agent": agent_id}
        else:
            # Handle in supervisor
            logger.info("Routing decision: supervisor (self-handle)")

            if message_type != "general":
                # Service unavailable message
                service_name = {
                    "url": "URL extraction",
                    "pdf": "PDF processing",
                    "notion": "Notion integration"
                }.get(message_type, "the requested service")

                error_msg = f"I detected that you want to use {service_name}, but this service is currently unavailable. Please check your configuration or contact support."
                return {
                    "current_agent": "supervisor",
                    "current_response": error_msg,
                    "messages": [AIMessage(content=error_msg)]
                }

            return {"current_agent": "supervisor"}

    async def _process_message(self, state: AgentState) -> Dict[str, Any]:
        """
        Process the user's message and generate a response.

        This is the main processing node. It takes the conversation history,
        adds a system prompt, and generates a response using the LLM.

        Note: state["messages"] already contains the full conversation history
        from the database (populated in process() method). We just prepend
        the system prompt for context.

        Args:
            state: Current agent state with messages and context

        Returns:
            Dict with state updates (current_response, messages)
        """
        logger.info("=== ENTERED _process_message node ===")
        logger.debug(f"State keys: {state.keys()}")
        logger.debug(f"Number of messages: {len(state.get('messages', []))}")

        # Extract conversation context
        context = state.get("conversation_context", {})
        user_name = context.get("user_name", "User")

        # Build system prompt from YAML configuration
        # self.system_prompt comes from BaseAgent._build_system_prompt()
        system_prompt_content = f"{self.system_prompt}\n\nYou are conversing with {user_name}."

        system_prompt = SystemMessage(content=system_prompt_content)

        # Combine system prompt with conversation history
        # state["messages"] already contains full conversation history from DB
        messages = [system_prompt] + state["messages"]

        # Generate response using LLM
        try:
            response = await self.llm.ainvoke(messages)  # Use ainvoke for async
            response_text = response.content

            logger.info(f"Generated response: {response_text[:100]}...")

            return {
                "current_response": response_text,
                "messages": [AIMessage(content=response_text)]
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            error_message = "I apologize, but I encountered an error processing your message. Please try again."

            return {
                "current_response": error_message,
                "messages": [AIMessage(content=error_message)]
            }


    def _route_to_node(self, state: AgentState) -> str:
        """
        Conditional edge function: Route to specific node based on current_agent.

        Returns:
            Node name: "url_extractor", "pdf_processor", "notion_agent", or "process"
        """
        current_agent = state.get("current_agent", "supervisor")

        if current_agent == "url_extractor":
            logger.info("Routing to url_extractor subgraph")
            return "url_extractor"
        elif current_agent == "pdf_processor":
            logger.info("Routing to pdf_processor subgraph")
            return "pdf_processor"
        elif current_agent == "notion_agent":
            logger.info("Routing to notion_agent subgraph")
            return "notion_agent"
        else:
            logger.info("Processing in supervisor")
            return "process"

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph with subgraph integration and checkpointing.

        Graph structure:
        START → detect_and_route → [url_extractor OR pdf_processor OR notion_agent OR process] → END

        Where:
        - detect_and_route: Detects message type and determines routing
        - url_extractor: URLExtractorAgent subgraph
        - pdf_processor: PDFProcessorAgent subgraph
        - notion_agent: NotionAgent subgraph
        - process: Handles message in supervisor (fallback)

        Returns:
            StateGraph: Compiled LangGraph graph with integrated subgraphs and checkpointing
        """
        # Create graph with AgentState
        workflow = StateGraph(AgentState)

        # Add supervisor nodes
        workflow.add_node("detect_and_route", self._detect_and_route)
        workflow.add_node("process", self._process_message)

        # Get agent registry and add subgraph nodes
        registry = AgentRegistry.get_instance()

        # Add URL Extractor subgraph
        if registry.is_agent_available("url_extractor"):
            url_agent = registry.get_agent("url_extractor")
            if hasattr(url_agent, 'subgraph'):
                workflow.add_node("url_extractor", url_agent.subgraph)
                logger.info("Added URLExtractorAgent subgraph to supervisor")
            else:
                logger.warning("URLExtractorAgent does not have subgraph attribute")

        # Add PDF Processor subgraph
        if registry.is_agent_available("pdf_processor"):
            pdf_agent = registry.get_agent("pdf_processor")
            if hasattr(pdf_agent, 'subgraph'):
                workflow.add_node("pdf_processor", pdf_agent.subgraph)
                logger.info("Added PDFProcessorAgent subgraph to supervisor")
            else:
                logger.warning("PDFProcessorAgent does not have subgraph attribute")

        # Add Notion Agent subgraph
        if registry.is_agent_available("notion_agent"):
            notion_agent = registry.get_agent("notion_agent")
            if hasattr(notion_agent, 'subgraph'):
                workflow.add_node("notion_agent", notion_agent.subgraph)
                logger.info("Added NotionAgent subgraph to supervisor")
            else:
                logger.warning("NotionAgent does not have subgraph attribute")

        # Set entry point
        workflow.set_entry_point("detect_and_route")

        # Build conditional routing map dynamically based on available agents
        routing_map = {"process": "process"}  # Always have process as fallback

        if registry.is_agent_available("url_extractor") and hasattr(registry.get_agent("url_extractor"), 'subgraph'):
            routing_map["url_extractor"] = "url_extractor"
            workflow.add_edge("url_extractor", END)

        if registry.is_agent_available("pdf_processor") and hasattr(registry.get_agent("pdf_processor"), 'subgraph'):
            routing_map["pdf_processor"] = "pdf_processor"
            workflow.add_edge("pdf_processor", END)

        if registry.is_agent_available("notion_agent") and hasattr(registry.get_agent("notion_agent"), 'subgraph'):
            routing_map["notion_agent"] = "notion_agent"
            workflow.add_edge("notion_agent", END)

        # Add conditional routing with only available agents
        workflow.add_conditional_edges(
            "detect_and_route",
            self._route_to_node,
            routing_map
        )

        # Process always leads to END
        workflow.add_edge("process", END)

        # Initialize in-memory checkpointer
        self.checkpointer = MemorySaver()

        # Compile graph with checkpointer
        graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("Supervisor graph with integrated subgraphs and checkpointing compiled successfully")
        return graph

    async def process(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None,
        user_context: Dict[str, Any] = None,
    ) -> str:
        """
        Process a user message and generate a response.

        This is the main entry point for the agent. It converts the message
        and history into LangChain format, runs the graph with checkpointing,
        and returns the response.

        Args:
            message: The user's message text
            conversation_history: Previous messages [{"role": "user/agent", "content": "..."}]
            user_context: Additional context about the user (name, preferences, etc.)
                         Should include 'conversation_id' for checkpointing

        Returns:
            str: The agent's response
        """
        logger.info("="*80)
        logger.info(f"=== SUPERVISOR AGENT PROCESS STARTED ===")
        logger.info(f"Message: {message[:100]}...")
        logger.info(f"User context: {user_context}")
        logger.info(f"History length: {len(conversation_history) if conversation_history else 0}")
        logger.info("="*80)

        # Convert conversation history to LangChain messages
        messages = []
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "agent":
                    messages.append(AIMessage(content=msg["content"]))

        # Add current message
        messages.append(HumanMessage(content=message))

        # Build initial state
        initial_state: AgentState = {
            "messages": messages,
            "conversation_context": user_context or {},
            "current_response": "",
            # Routing state
            "current_agent": "supervisor",
            "routed_data": {},
            "routing_history": [],
        }
        
        # Use conversation_id from user_context for checkpointing
        conversation_id = user_context.get("conversation_id", "default") if user_context else "default"

        # Configure with thread_id for checkpointing
        config = {
            "configurable": {
                "thread_id": conversation_id
            }
        }

        # Run the graph with checkpointing
        try:
            logger.info("=== ABOUT TO INVOKE GRAPH ===")
            logger.info(f"Initial state: {initial_state.keys()}")
            logger.info(f"Config: {config}")

            # Use ainvoke for async graph execution with config
            final_state = await self.graph.ainvoke(initial_state, config=config)

            logger.info("=== GRAPH EXECUTION COMPLETE ===")
            logger.info(f"Final state keys: {final_state.keys()}")
            logger.info(f"Current agent: {final_state.get('current_agent', 'NONE')}")
            logger.info(f"Current response: '{final_state.get('current_response', 'EMPTY')}'")
            logger.info(f"Routing history: {final_state.get('routing_history', [])}")

            response = final_state["current_response"]

            logger.info(f"=== RETURNING RESPONSE: {response[:100] if response else 'EMPTY'} ===")
            logger.info("="*80)
            return response

        except Exception as e:
            logger.error(f"Error in agent processing: {e}", exc_info=True)
            return "I apologize, but I encountered an error. Please try again."
