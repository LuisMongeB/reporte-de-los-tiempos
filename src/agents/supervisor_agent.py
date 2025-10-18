"""
Supervisor agent for processing user messages and generating responses.
"""

import logging
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.core.config import get_config
from src.agents.BaseAgent import BaseAgent

logger = logging.getLogger(__name__)
settings = get_config()


class AgentState(TypedDict):
    """
    State maintained throughout the agent's execution.

    LangGraph passes this state between nodes. Each node can read and modify it.
    """

    messages: List  # Conversation history
    conversation_context: Dict[str, Any]  # Additional context (user info, etc.)
    current_response: str  # The agent's response


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

    def _process_message(self, state: AgentState) -> AgentState:
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

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph.

        For MVP, this is a simple linear graph:
        START -> process_message -> END

        In future phases, this will branch to different specialized agents.

        Returns:
            StateGraph: Compiled LangGraph graph
        """
        # Create graph with AgentState
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("process_message", self._process_message)

        # Define edges
        workflow.set_entry_point("process_message")
        workflow.add_edge("process_message", END)

        # Compile graph
        graph = workflow.compile()

        logger.info("Agent graph compiled successfully")
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
        }

        # Run the graph
        try:
            # LangGraph's invoke is synchronous in 0.6.8
            final_state = self.graph.invoke(initial_state)
            response = final_state["current_response"]

            logger.info("Message processing complete")
            return response

        except Exception as e:
            logger.error(f"Error in agent processing: {e}", exc_info=True)
            return "I apologize, but I encountered an error. Please try again."
