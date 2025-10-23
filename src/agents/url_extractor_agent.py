"""
URLExtractorAgent - Specialized agent for handling URL extraction and analysis

This agent uses Tavily Extract API to fetch and analyze content from URLs.
It presents users with options to either extract and analyze content or save
URLs to Notion as bookmarks.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.BaseAgent import BaseAgent
from src.core.config import get_config
from src.tools.tavily_tools import TavilyTools

logger = logging.getLogger(__name__)
settings = get_config()


class URLExtractorAgent(BaseAgent):
    """
    Specialized agent for extracting and analyzing URL content.

    Workflow:
    1. User sends URL → Agent presents options (extract or save to Notion)
    2. If extract → Request approval → Extract → Analyze → Offer to save
    3. If save → Route to NotionAgent

    Uses Tavily Extract API for clean content extraction.
    """

    def __init__(self, prompt_file: str = "specialized/url_extractor_agent.yaml"):
        """
        Initialize the URL extractor agent.

        Args:
            prompt_file: Path to YAML prompt configuration
        """
        logger.info("Initializing URLExtractorAgent...")

        # Initialize Tavily tools BEFORE calling super().__init__()
        # because BaseAgent.__init__ calls _register_tools() which needs self.tavily_tools
        self.tavily_tools = None
        if settings.tavily_api_key:
            try:
                self.tavily_tools = TavilyTools()
                logger.info("TavilyTools initialized successfully")
            except ValueError as e:
                logger.warning(f"TavilyTools not available: {e}")
        else:
            logger.warning("Tavily API key not configured - extraction will not work")

        # Initialize BaseAgent (loads prompt, validates, builds system prompt)
        # This will call _register_tools() which needs self.tavily_tools to be set
        super().__init__(prompt_file)

        # Initialize LLM for content analysis
        self.llm = self._initialize_llm()

        logger.info("URLExtractorAgent initialized successfully")

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
        parts.append(f"**Tone:** {config.get('tone', 'helpful and conversational')}")
        parts.append("")
        parts.append("**Instructions:**")
        parts.append(config.get("instructions", ""))

        return "\n".join(parts)

    def _register_tools(self) -> List[Any]:
        """
        Register tools available to the URL extractor agent.

        Returns:
            List of tool objects
        """
        tools = []

        # Check if tavily_tools has been initialized (it's set before super().__init__)
        if hasattr(self, 'tavily_tools') and self.tavily_tools:
            tools.append(self.tavily_tools)

        return tools

    async def process_message(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an incoming message.

        Handles the complete URL extraction workflow:
        1. Detect URL in message
        2. Present options (extract or save)
        3. Handle user choice
        4. Execute action with approval

        Args:
            message: The user's message
            context: Conversation context including:
                - conversation_id: Unique conversation identifier
                - user_id: User identifier
                - conversation_state: Agent-specific state (workflow step, URL, etc.)
                - approval_granted: Whether user approved the action

        Returns:
            Dictionary containing:
            - status: "complete", "awaiting_input", "awaiting_approval", "error", "routing"
            - response: The agent's response message
            - should_stream: Whether to stream the response
            - metadata: Optional metadata
            - next_agent: (if status="routing") Next agent to route to
            - conversation_state: Updated state for next interaction
        """
        logger.info(f"URLExtractorAgent processing message: {message[:100]}...")

        # Get current conversation state
        state = context.get("conversation_state", {})
        workflow_step = state.get("step", "detect_url")
        stored_url = state.get("url")
        user_choice = state.get("user_choice")

        # Step 1: Detect URL in message
        if workflow_step == "detect_url":
            url = self._extract_url(message)

            if not url:
                return {
                    "status": "error",
                    "response": "I couldn't find a valid URL in your message. Please send a URL starting with http:// or https://",
                    "should_stream": False,
                    "conversation_state": {}
                }

            # Present options to user
            flow = self.prompt_config["interaction_flow"]["step_1_present_options"]
            response = flow["message"].format(url=url)

            return {
                "status": "awaiting_input",
                "response": response,
                "should_stream": False,
                "conversation_state": {
                    "step": "awaiting_choice",
                    "url": url
                }
            }

        # Step 2: Process user choice
        if workflow_step == "awaiting_choice":
            choice = message.strip()

            if choice not in ["1", "2"]:
                return {
                    "status": "awaiting_input",
                    "response": "Please reply with 1 (to extract and analyze) or 2 (to save to Notion).",
                    "should_stream": False,
                    "conversation_state": state  # Keep same state
                }

            if choice == "1":
                # User wants to extract content
                # Check if Tavily is configured
                if not self.tavily_tools:
                    error_msg = self.get_error_message("tavily_not_configured")
                    return {
                        "status": "error",
                        "response": error_msg,
                        "should_stream": False,
                        "conversation_state": {}
                    }

                # Request approval for extraction
                approval_msg = self.get_approval_message("tavily_extract", url=stored_url)

                return {
                    "status": "awaiting_approval",
                    "response": approval_msg,
                    "should_stream": False,
                    "conversation_state": {
                        "step": "extract_approved",
                        "url": stored_url,
                        "user_choice": "extract"
                    }
                }

            else:  # choice == "2"
                # User wants to save to Notion (route to NotionAgent)
                # This will be implemented in Sprint 2.3
                return {
                    "status": "routing",
                    "response": "I'll save this URL to Notion...",
                    "should_stream": False,
                    "next_agent": "notion_agent",
                    "metadata": {
                        "url": stored_url,
                        "action": "save_url"
                    },
                    "conversation_state": {}
                }

        # Step 3: Extract and analyze content (after approval)
        if workflow_step == "extract_approved":
            # Check if approval was granted
            if not context.get("approval_granted"):
                return {
                    "status": "complete",
                    "response": "Okay, I won't extract the content. Let me know if you need anything else!",
                    "should_stream": False,
                    "conversation_state": {}
                }

            # Extract content using Tavily
            result = await self._extract_and_analyze(stored_url, context)

            return result

        # Unknown state - reset
        return {
            "status": "error",
            "response": "I'm sorry, I lost track of our conversation. Please send the URL again.",
            "should_stream": False,
            "conversation_state": {}
        }

    def _initialize_llm(self):
        """
        Initialize the OpenAI LLM for content analysis.

        Returns:
            ChatOpenAI: Configured LLM instance
        """
        logger.info(f"Initializing OpenAI model for URL analysis: {settings.llm_model}")
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            max_tokens=settings.llm_max_tokens,
        )

    def _extract_url(self, message: str) -> Optional[str]:
        """
        Extract URL from message using regex.

        Args:
            message: User's message

        Returns:
            Extracted URL or None if no URL found
        """
        # Simple URL regex pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, message)

        if match:
            url = match.group(0)
            logger.info(f"Extracted URL: {url}")
            return url

        return None

    async def _extract_and_analyze(
        self,
        url: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract content from URL and generate analysis using LLM.

        Args:
            url: URL to extract content from
            context: Conversation context

        Returns:
            Response dictionary with analysis
        """
        logger.info(f"Extracting and analyzing content from: {url}")

        # Extract content using Tavily
        try:
            extraction_result = await self.tavily_tools.extract_url(url)

            if not extraction_result['success']:
                error_msg = self.get_error_message(
                    "extraction_failed",
                    error=extraction_result.get('error', 'Unknown error')
                )
                return {
                    "status": "error",
                    "response": error_msg,
                    "should_stream": False,
                    "conversation_state": {}
                }

            # Got content - analyze it with LLM
            title = extraction_result['title']
            content = extraction_result['content']
            word_count = extraction_result['word_count']

            logger.info(f"Extracted {word_count} words from {url}")

            # Build analysis prompt
            analysis_prompt = f"""
I've extracted the following content from {url}:

**Title:** {title}
**Word Count:** {word_count}

**Content:**
{content[:3000]}{'...' if len(content) > 3000 else ''}

Please provide a concise analysis with:
1. A brief summary (2-3 sentences)
2. Key points or takeaways (3-5 bullet points)
3. Main topics covered

Keep your response clear and well-formatted.
"""

            # Generate analysis using LLM
            system_prompt = SystemMessage(content=self.system_prompt)
            user_message = HumanMessage(content=analysis_prompt)

            try:
                response = self.llm.invoke([system_prompt, user_message])
                analysis = response.content

                # Add post-extraction offer to save to Notion
                flow = self.prompt_config["interaction_flow"]["step_4_post_extraction"]
                post_message = flow["message"]

                final_response = f"""✅ **Content Extraction Complete**

**URL:** {url}
**Title:** {title}
**Word Count:** {word_count}

{analysis}

---

{post_message}
"""

                return {
                    "status": "complete",
                    "response": final_response,
                    "should_stream": True,  # Stream the analysis for better UX
                    "metadata": {
                        "url": url,
                        "title": title,
                        "word_count": word_count,
                        "extraction_success": True
                    },
                    "conversation_state": {
                        "step": "offer_notion_save",
                        "url": url,
                        "title": title,
                        "content": content
                    }
                }

            except Exception as e:
                logger.error(f"Error analyzing content: {e}", exc_info=True)
                return {
                    "status": "error",
                    "response": f"I extracted the content but encountered an error during analysis: {str(e)}",
                    "should_stream": False,
                    "conversation_state": {}
                }

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}", exc_info=True)
            error_msg = self.get_error_message("extraction_failed", error=str(e))
            return {
                "status": "error",
                "response": error_msg,
                "should_stream": False,
                "conversation_state": {}
            }
