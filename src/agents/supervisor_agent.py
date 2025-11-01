"""
Supervisor agent for processing user messages and generating responses.
"""

import logging
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.core.config import get_config
from src.agents.BaseAgent import BaseAgent
from src.agents.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)
settings = get_config()


class AgentState(TypedDict):
    """
    State maintained throughout the agent's execution.

    LangGraph passes this state between nodes. Each node can read and modify it.
    This state supports multi-agent routing and orchestration.
    """

    # Core conversation state
    messages: List  # Full conversation history (includes specialized agent responses)
    conversation_context: Dict[str, Any]  # Additional context (user info, etc.)
    current_response: str  # The agent's current response being built

    # Routing state (for multi-agent orchestration)
    current_agent: str  # Which agent is currently handling the message (default: "supervisor")
    routed_data: Dict[str, Any]  # Data passed between agents (e.g., extracted content, URLs)
    routing_history: List[Dict[str, str]]  # Track which agents were involved in processing


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

    def _detect_message_type(self, message: str) -> str:
        """
        Detect the type of message to determine routing.

        Uses routing patterns from YAML configuration to identify:
        - URLs (for URLExtractorAgent)
        - PDFs (for PDFProcessorAgent)
        - Notion keywords (for NotionAgent)
        - General messages (handled by supervisor)

        Args:
            message: The user's message

        Returns:
            Message type: "url", "pdf", "notion", or "general"
        """
        routing_config = self.prompt_config.get("routing", {})
        patterns = routing_config.get("detection_patterns", {})

        # Check URL pattern
        url_pattern = patterns.get("url_pattern", r"^https?://")
        if re.match(url_pattern, message.strip()):
            logger.info(f"Detected URL message: {message[:50]}...")
            return "url"

        # Check PDF pattern
        pdf_pattern = patterns.get("pdf_pattern", r".*\.pdf$")
        if re.match(pdf_pattern, message.strip(), re.IGNORECASE):
            logger.info(f"Detected PDF message: {message[:50]}...")
            return "pdf"

        # Check Notion keywords
        notion_keywords = patterns.get("notion_keywords", [])
        message_lower = message.lower()
        if any(keyword.lower() in message_lower for keyword in notion_keywords):
            logger.info(f"Detected Notion keyword in message: {message[:50]}...")
            return "notion"

        logger.debug(f"Message classified as general: {message[:50]}...")
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

    async def _route_to_agent(
        self,
        agent_id: str,
        message: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """
        Route a message to a specialized agent.

        This method handles the full routing workflow:
        1. Check agent availability
        2. Get agent instance from registry
        3. Prepare context with routing data
        4. Call agent's process_message()
        5. Update routing history
        6. Handle errors gracefully

        Args:
            agent_id: The ID of the agent to route to
            message: The message to process
            state: Current agent state (contains routing_history, routed_data, etc.)

        Returns:
            Agent response dict with:
            - status: "complete", "awaiting_input", "routing", or "error"
            - response: Agent's response message
            - metadata: Optional metadata for routing
            - conversation_state: State to persist for multi-step workflows
        """
        try:
            registry = AgentRegistry.get_instance()

            # Check if agent is available
            if not registry.is_agent_available(agent_id):
                logger.warning(f"Agent {agent_id} is not available")
                agent_info = registry.get_agent_info(agent_id) if agent_id in registry._metadata else {}
                agent_name = agent_info.get("name", agent_id)

                return {
                    "status": "error",
                    "response": f"The {agent_name} service is currently unavailable. Please check your configuration.",
                    "conversation_state": {}
                }

            # Get agent instance
            agent = registry.get_agent(agent_id)

            # Prepare context for specialized agent
            # Provide both `metadata` and `routed_data` keys for compatibility
            # with specialized agents (some expect `metadata`, others `routed_data`).
            # Also forward any agent-specific conversation_state.
            context = {
                "conversation_context": state.get("conversation_context", {}),
                "metadata": state.get("routed_data", {}),
                "routed_data": state.get("routed_data", {}),
                "conversation_state": state.get("conversation_state", {}),
            }

            # Call specialized agent
            logger.info(f"Routing to agent: {agent_id}")
            result = await agent.process_message(message, context)

            # Update routing history
            state["routing_history"].append({
                "agent": agent_id,
                "message": message[:100],  # Truncate for logging
                "status": result.get("status", "unknown")
            })

            # Update current agent
            state["current_agent"] = agent_id

            logger.info(f"Agent {agent_id} responded with status: {result.get('status')}")
            return result

        except Exception as e:
            logger.error(f"Error routing to {agent_id}: {e}", exc_info=True)
            return {
                "status": "error",
                "response": "I encountered an unexpected error while processing your request. Please try again.",
                "conversation_state": {}
            }

    async def _detect_and_route(self, state: AgentState) -> AgentState:
        """
        LangGraph node: Detect message type and route to appropriate agent.

        This is the routing node in the LangGraph workflow. It:
        1. Extracts the latest user message
        2. Detects message type
        3. Determines which agent should handle it
        4. Routes to specialized agent OR continues to supervisor processing

        The node updates state["current_agent"] to indicate routing decision.

        Args:
            state: Current agent state

        Returns:
            Updated state with routing decision
        """
        # Extract latest message (last message in the list)
        if not state["messages"]:
            logger.warning("No messages in state for routing detection")
            state["current_agent"] = "supervisor"
            return state

        latest_message = state["messages"][-1]
        message_content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)

        # Detect message type
        message_type = self._detect_message_type(message_content)

        # Determine routing
        agent_id = self._determine_routing(message_type, message_content)

        if agent_id:
            # Mark for routing to specialized agent
            state["current_agent"] = agent_id
            logger.info(f"Routing decision: {agent_id}")
        else:
            # Handle in supervisor
            state["current_agent"] = "supervisor"
            if message_type != "general":
                # Service unavailable message
                service_name = {
                    "url": "URL extraction",
                    "pdf": "PDF processing",
                    "notion": "Notion integration"
                }.get(message_type, "the requested service")

                error_msg = f"I detected that you want to use {service_name}, but this service is currently unavailable. Please check your configuration or contact support."
                state["current_response"] = error_msg
                state["messages"].append(AIMessage(content=error_msg))

            logger.info("Routing decision: supervisor (self-handle)")

        return state

    async def _process_message(self, state: AgentState) -> AgentState:
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
            AgentState: Updated state with agent's response
        """
        logger.info("Processing message in supervisor agent...")

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
            response = self.llm.invoke(messages)
            response_text = response.content

            logger.info(f"Generated response: {response_text[:100]}...")

            # Update state with response
            state["current_response"] = response_text
            state["messages"].append(AIMessage(content=response_text))

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            error_message = "I apologize, but I encountered an error processing your message. Please try again."
            state["current_response"] = error_message
            state["messages"].append(AIMessage(content=error_message))

        return state

    async def _handle_routed_message(self, state: AgentState) -> AgentState:
        """
        LangGraph node: Handle message routing to specialized agents.

        This node executes when state["current_agent"] != "supervisor".
        It routes the message to the specified agent and handles the response.

        Supports multi-step routing (e.g., URLExtractor → NotionAgent).

        Args:
            state: Current agent state

        Returns:
            Updated state with agent's response
        """
        agent_id = state["current_agent"]
        logger.info(f"Handling routed message for agent: {agent_id}")

        # Extract the latest user message
        latest_message = state["messages"][-1]
        message_content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)

        # Route to the specialized agent
        result = await self._route_to_agent(agent_id, message_content, state)

        # Handle multi-step routing
        if result.get("status") == "routing":
            next_agent = result.get("next_agent")
            if next_agent:
                logger.info(f"Agent {agent_id} requesting handoff to {next_agent}")
                # Update routed_data with metadata from first agent
                state["routed_data"] = result.get("metadata", {})
                # Route to next agent
                result = await self._route_to_agent(next_agent, message_content, state)

        # Update state with agent's response
        response_text = result.get("response", "")
        state["current_response"] = response_text

        # Add agent response to conversation history
        if response_text:
            state["messages"].append(AIMessage(content=response_text))

        return state

    def _should_route(self, state: AgentState) -> str:
        """
        Conditional edge function: Determine if message should be routed.

        Returns:
            "route" if should route to specialized agent
            "process" if should handle in supervisor
        """
        if state["current_agent"] != "supervisor":
            return "route"
        return "process"

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph with routing support.

        Graph structure:
        START → detect_and_route → [route OR process] → END

        Where:
        - detect_and_route: Detects message type and determines routing
        - route: Handles routing to specialized agents
        - process: Handles message in supervisor (fallback)

        Returns:
            StateGraph: Compiled LangGraph graph
        """
        # Create graph with AgentState
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("detect_and_route", self._detect_and_route)
        workflow.add_node("route", self._handle_routed_message)
        workflow.add_node("process", self._process_message)

        # Set entry point
        workflow.set_entry_point("detect_and_route")

        # Add conditional routing from detect_and_route
        workflow.add_conditional_edges(
            "detect_and_route",
            self._should_route,
            {
                "route": "route",
                "process": "process"
            }
        )

        # Both route and process lead to END
        workflow.add_edge("route", END)
        workflow.add_edge("process", END)

        # Compile graph
        graph = workflow.compile()

        logger.info("Agent graph with routing compiled successfully")
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
        and history into LangChain format, runs the graph, and returns the response.

        Args:
            message: The user's message text
            conversation_history: Previous messages [{"role": "user/agent", "content": "..."}]
            user_context: Additional context about the user (name, preferences, etc.)

        Returns:
            str: The agent's response
        """
        logger.info(f"Processing message: {message[:100]}...")

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

        # Run the graph
        try:
            # Use ainvoke for async graph execution (supports async nodes)
            final_state = await self.graph.ainvoke(initial_state)
            response = final_state["current_response"]

            logger.info("Message processing complete")
            return response

        except Exception as e:
            logger.error(f"Error in agent processing: {e}", exc_info=True)
            return "I apologize, but I encountered an error. Please try again."
