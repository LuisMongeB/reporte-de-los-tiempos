"""
PDFProcessorAgent - Specialized agent for processing PDF files

This agent handles PDF files by extracting text using pypdf or processing
with Vision LLM for scanned documents. It presents users with options to
choose the extraction method or save the PDF summary to Notion.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from operator import add

from src.agents.BaseAgent import BaseAgent
from src.core.config import get_config
from src.tools.pdf_processor_tools import PDFProcessorTools

logger = logging.getLogger(__name__)
settings = get_config()


class PDFProcessorState(TypedDict):
    """State for the PDF processor subgraph."""
    # Core state (compatible with AgentState)
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_context: Dict[str, Any]
    current_response: str

    # PDF processing specific
    file_path: str
    file_name: str
    file_size: int
    user_choice: str  # "text", "vision", or "save"
    extracted_content: str
    analysis: str
    workflow_step: str

    # Metadata
    page_count: int
    word_count: int
    extraction_method: str


class PDFProcessorAgent(BaseAgent):
    """
    Specialized agent for processing PDF files.

    Workflow:
    1. User sends PDF file → Agent downloads and presents options
    2. Options: text extraction (pypdf), vision LLM, or save to Notion
    3. If text/vision → Request approval → Extract → Analyze → Offer to save
    4. If save → Route to NotionAgent

    Supports both text-based and scanned PDFs.
    """

    def __init__(self, prompt_file: str = "specialized/pdf_processor_agent.yaml"):
        """
        Initialize the PDF processor agent.

        Args:
            prompt_file: Path to YAML prompt configuration
        """
        logger.info("Initializing PDFProcessorAgent...")

        # Initialize PDF tools BEFORE calling super().__init__()
        self.pdf_tools = PDFProcessorTools()

        # Initialize BaseAgent (loads prompt, validates, builds system prompt)
        super().__init__(prompt_file)

        # Initialize LLM for content analysis
        self.llm = self._initialize_llm()

        # Build LangGraph subgraph
        self.subgraph = self._build_subgraph()

        logger.info("PDFProcessorAgent initialized successfully with LangGraph subgraph")

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
        parts.append(f"**Tone:** {config.get('tone', 'helpful and professional')}")
        parts.append("")
        parts.append("**Instructions:**")
        parts.append(config.get("instructions", ""))

        return "\n".join(parts)

    def _register_tools(self) -> List[Any]:
        """
        Register tools available to the PDF processor agent.

        Returns:
            List of tool objects
        """
        tools = []

        if hasattr(self, 'pdf_tools') and self.pdf_tools:
            tools.append(self.pdf_tools)

        return tools

    def _build_subgraph(self) -> StateGraph:
        """
        Build the LangGraph subgraph for PDF processing workflow.

        Workflow:
        1. receive_file -> present_options
        2. present_options -> await_choice
        3. await_choice -> [extract_text, vision_extract, save_to_notion]
        4. extract_text -> analyze_content -> offer_notion_save
        5. vision_extract -> analyze_content -> offer_notion_save
        6. offer_notion_save -> END
        """
        workflow = StateGraph(PDFProcessorState)

        # Add nodes for each workflow step
        workflow.add_node("receive_file", self._receive_file_node)
        workflow.add_node("present_options", self._present_options_node)
        workflow.add_node("await_choice", self._await_choice_node)
        workflow.add_node("extract_text", self._extract_text_node)
        workflow.add_node("vision_extract", self._vision_extract_node)
        workflow.add_node("analyze_content", self._analyze_content_node)
        workflow.add_node("offer_notion_save", self._offer_notion_save_node)

        # Define workflow edges
        workflow.set_entry_point("receive_file")
        workflow.add_edge("receive_file", "present_options")
        workflow.add_edge("present_options", "await_choice")

        # Conditional routing after user choice
        workflow.add_conditional_edges(
            "await_choice",
            self._route_after_choice,
            {
                "extract_text": "extract_text",
                "vision_extract": "vision_extract",
                "save_to_notion": END,
                "invalid": "await_choice"
            }
        )

        workflow.add_edge("extract_text", "analyze_content")
        workflow.add_edge("vision_extract", "analyze_content")
        workflow.add_edge("analyze_content", "offer_notion_save")
        workflow.add_edge("offer_notion_save", END)

        # Compile with checkpointing and interrupts for human approval
        checkpointer = MemorySaver()
        graph = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["extract_text", "vision_extract"]
        )

        logger.info("PDFProcessorAgent subgraph compiled successfully")
        return graph

    def _receive_file_node(self, state: PDFProcessorState) -> Dict[str, Any]:
        """
        Node: Validate and store file information.

        Returns updates to state.
        """
        logger.info("receive_file_node: Validating PDF file")

        file_path = state.get("file_path")
        file_name = state.get("file_name", "document.pdf")
        file_size = state.get("file_size", 0)

        # Validation checks
        if not file_path:
            error_msg = "No PDF file received. Please send a PDF file."
            return {
                "current_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "workflow_step": "error"
            }

        if not Path(file_path).exists():
            error_msg = self.get_error_message("invalid_pdf")
            return {
                "current_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "workflow_step": "error"
            }

        # Check file size
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            size_mb = file_size / (1024 * 1024)
            error_msg = self.get_error_message("file_too_large", size=f"{size_mb:.1f}MB")
            return {
                "current_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "workflow_step": "error"
            }

        logger.info(f"File validated: {file_name} ({self._format_file_size(file_size)})")

        return {
            "workflow_step": "present_options",
            "file_path": file_path,
            "file_name": file_name,
            "file_size": file_size
        }

    def _present_options_node(self, state: PDFProcessorState) -> Dict[str, Any]:
        """
        Node: Present extraction method options to user.

        Returns updates to state.
        """
        logger.info("present_options_node: Presenting extraction options")

        file_name = state.get("file_name", "document.pdf")
        file_size = state.get("file_size", 0)

        flow = self.prompt_config["interaction_flow"]["step_2_present_options"]
        file_size_str = self._format_file_size(file_size)
        response = flow["message"].format(filename=file_name, file_size=file_size_str)

        return {
            "current_response": response,
            "messages": [AIMessage(content=response)],
            "workflow_step": "await_choice"
        }

    def _await_choice_node(self, state: PDFProcessorState) -> Dict[str, Any]:
        """
        Node: Wait for and validate user choice.

        Returns updates to state.
        """
        logger.info("await_choice_node: Processing user choice")

        # Get the last user message
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, HumanMessage):
                choice = last_msg.content.strip()

                if choice in ["1", "2", "3"]:
                    # Valid choice - update state
                    choice_map = {"1": "text", "2": "vision", "3": "save"}
                    return {
                        "user_choice": choice_map[choice],
                        "workflow_step": "choice_made"
                    }

        # Invalid choice - ask again
        error_msg = "Please reply with 1 (text extraction), 2 (Vision LLM), or 3 (save to Notion)."
        return {
            "current_response": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "workflow_step": "await_choice"
        }

    def _route_after_choice(self, state: PDFProcessorState) -> Literal["extract_text", "vision_extract", "save_to_notion", "invalid"]:
        """
        Conditional edge: Route based on user choice.

        Returns:
            Next node name
        """
        user_choice = state.get("user_choice", "")
        logger.info(f"Routing after choice: {user_choice}")

        if user_choice == "text":
            return "extract_text"
        elif user_choice == "vision":
            return "vision_extract"
        elif user_choice == "save":
            return "save_to_notion"
        else:
            return "invalid"

    async def _extract_text_node(self, state: PDFProcessorState) -> Dict[str, Any]:
        """
        Node: Extract text using pypdf.

        Returns updates to state.
        """
        logger.info("extract_text_node: Extracting text with pypdf")

        file_path = state.get("file_path")
        file_name = state.get("file_name", "document.pdf")

        try:
            extraction_result = await self.pdf_tools.extract_text_pypdf(file_path)

            if not extraction_result['success']:
                error_msg = self.get_error_message(
                    "extraction_failed",
                    error=extraction_result.get('error', 'Unknown error')
                )
                return {
                    "current_response": error_msg,
                    "messages": [AIMessage(content=error_msg)],
                    "workflow_step": "error"
                }

            text = extraction_result['text']
            page_count = extraction_result['page_count']
            word_count = extraction_result['word_count']

            logger.info(f"Extracted {word_count} words from {page_count} pages")

            return {
                "extracted_content": text,
                "page_count": page_count,
                "word_count": word_count,
                "extraction_method": "pypdf",
                "workflow_step": "analyze"
            }

        except Exception as e:
            logger.error(f"Error extracting text: {e}", exc_info=True)
            error_msg = self.get_error_message("extraction_failed", error=str(e))
            return {
                "current_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "workflow_step": "error"
            }

    async def _vision_extract_node(self, state: PDFProcessorState) -> Dict[str, Any]:
        """
        Node: Extract content using Vision LLM.

        Returns updates to state.
        """
        logger.info("vision_extract_node: Processing with Vision LLM")

        file_path = state.get("file_path")
        file_name = state.get("file_name", "document.pdf")

        try:
            vision_result = await self.pdf_tools.extract_with_vision(
                file_path,
                max_pages=settings.max_pdf_pages
            )

            if not vision_result['success']:
                error_msg = self.get_error_message(
                    "vision_failed",
                    error=vision_result.get('error', 'Unknown error')
                )
                return {
                    "current_response": error_msg,
                    "messages": [AIMessage(content=error_msg)],
                    "workflow_step": "error"
                }

            content = vision_result['content']
            pages_processed = vision_result['pages_processed']
            word_count = vision_result['word_count']

            logger.info(f"Vision LLM processed {pages_processed} pages, extracted {word_count} words")

            return {
                "extracted_content": content,
                "page_count": pages_processed,
                "word_count": word_count,
                "extraction_method": "vision_llm",
                "workflow_step": "analyze"
            }

        except Exception as e:
            logger.error(f"Error with Vision LLM: {e}", exc_info=True)
            error_msg = self.get_error_message("vision_failed", error=str(e))
            return {
                "current_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "workflow_step": "error"
            }

    async def _analyze_content_node(self, state: PDFProcessorState) -> Dict[str, Any]:
        """
        Node: Analyze extracted content with LLM.

        Returns updates to state.
        """
        logger.info("analyze_content_node: Analyzing extracted content")

        file_name = state.get("file_name", "document.pdf")
        content = state.get("extracted_content", "")
        page_count = state.get("page_count", 0)
        word_count = state.get("word_count", 0)
        extraction_method = state.get("extraction_method", "unknown")

        if not content:
            error_msg = "No content to analyze."
            return {
                "current_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "workflow_step": "error"
            }

        try:
            # Build analysis prompt
            analysis_prompt = f"""
            I've extracted text from a PDF file: {file_name}

            **Pages:** {page_count}
            **Word Count:** {word_count}

            **Content:**
            {content[:5000]}{'...' if len(content) > 5000 else ''}

            Please provide a concise analysis with:
            1. Document type and purpose (2-3 sentences)
            2. Key sections or topics covered (bullet points)
            3. Main takeaways or conclusions (3-5 points)

            Keep your response clear and well-formatted.
            """

            system_prompt = SystemMessage(content=self.system_prompt)
            user_message = HumanMessage(content=analysis_prompt)

            response = await self.llm.ainvoke([system_prompt, user_message])
            analysis = response.content

            logger.info("Content analysis complete")

            return {
                "analysis": analysis,
                "workflow_step": "offer_save"
            }

        except Exception as e:
            logger.error(f"Error analyzing content: {e}", exc_info=True)
            error_msg = f"I extracted the content but encountered an error during analysis: {str(e)}"
            return {
                "current_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "workflow_step": "error"
            }

    def _offer_notion_save_node(self, state: PDFProcessorState) -> Dict[str, Any]:
        """
        Node: Present final results and offer to save to Notion.

        Returns updates to state.
        """
        logger.info("offer_notion_save_node: Presenting results")

        file_name = state.get("file_name", "document.pdf")
        page_count = state.get("page_count", 0)
        word_count = state.get("word_count", 0)
        extraction_method = state.get("extraction_method", "unknown")
        analysis = state.get("analysis", "")

        # Build final response
        method_display = "pypdf" if extraction_method == "pypdf" else "Vision LLM (gpt-4o)"

        flow = self.prompt_config["interaction_flow"]["step_7_post_extraction"]
        post_message = flow["message"]

        final_response = f"""✅ **PDF Processing Complete**

        **File:** {file_name}
        **Pages:** {page_count}
        **Word Count:** {word_count}
        **Method:** {method_display}

        {analysis}

        ---

        {post_message}
        """

        # Cleanup file after processing
        file_path = state.get("file_path")
        if file_path:
            self.pdf_tools.cleanup_file(file_path)

        return {
            "current_response": final_response,
            "messages": [AIMessage(content=final_response)],
            "workflow_step": "complete"
        }

    async def process_message(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an incoming message with PDF file.

        Handles the complete PDF processing workflow:
        1. Receive PDF file path from context
        2. Present options (text extract, vision, or save)
        3. Handle user choice
        4. Execute action with approval

        Args:
            message: The user's message
            context: Conversation context including:
                - conversation_id: Unique conversation identifier
                - user_id: User identifier
                - conversation_state: Agent-specific state
                - approval_granted: Whether user approved the action
                - file_path: Path to the downloaded PDF file (set by handler)
                - file_name: Original filename
                - file_size: File size in bytes

        Returns:
            Dictionary containing:
            - status: "complete", "awaiting_input", "awaiting_approval", "error", "routing"
            - response: The agent's response message
            - should_stream: Whether to stream the response
            - metadata: Optional metadata
            - next_agent: (if status="routing") Next agent to route to
            - conversation_state: Updated state for next interaction
        """
        logger.info(f"PDFProcessorAgent processing message: {message[:100]}...")

        # Get current conversation state
        state = context.get("conversation_state", {})
        workflow_step = state.get("step", "receive_file")
        stored_file_path = state.get("file_path")
        stored_file_name = state.get("file_name")
        user_choice = state.get("user_choice")

        # Step 1: Receive PDF file
        if workflow_step == "receive_file":
            # Get file info from context (set by Telegram handler)
            file_path = context.get("file_path")
            file_name = context.get("file_name", "document.pdf")
            file_size = context.get("file_size", 0)

            if not file_path:
                return {
                    "status": "error",
                    "response": "No PDF file received. Please send a PDF file.",
                    "should_stream": False,
                    "conversation_state": {}
                }

            # Verify file exists
            if not Path(file_path).exists():
                return {
                    "status": "error",
                    "response": self.get_error_message("invalid_pdf"),
                    "should_stream": False,
                    "conversation_state": {}
                }

            # Check file size
            max_size_bytes = settings.max_file_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                size_mb = file_size / (1024 * 1024)
                error_msg = self.get_error_message("file_too_large", size=f"{size_mb:.1f}MB")
                return {
                    "status": "error",
                    "response": error_msg,
                    "should_stream": False,
                    "conversation_state": {}
                }

            # Present options to user
            flow = self.prompt_config["interaction_flow"]["step_2_present_options"]
            file_size_str = self._format_file_size(file_size)
            response = flow["message"].format(filename=file_name, file_size=file_size_str)

            return {
                "status": "awaiting_input",
                "response": response,
                "should_stream": False,
                "conversation_state": {
                    "step": "awaiting_choice",
                    "file_path": file_path,
                    "file_name": file_name,
                    "file_size": file_size
                }
            }

        # Step 2: Process user choice
        if workflow_step == "awaiting_choice":
            choice = message.strip()

            if choice not in ["1", "2", "3"]:
                return {
                    "status": "awaiting_input",
                    "response": "Please reply with 1 (text extraction), 2 (Vision LLM), or 3 (save to Notion).",
                    "should_stream": False,
                    "conversation_state": state
                }

            if choice == "1":
                # User wants text extraction
                approval_msg = self.get_approval_message(
                    "extract_text_pypdf",
                    filename=stored_file_name
                )

                return {
                    "status": "awaiting_approval",
                    "response": approval_msg,
                    "should_stream": False,
                    "conversation_state": {
                        "step": "text_extraction_approved",
                        "file_path": stored_file_path,
                        "file_name": stored_file_name,
                        "user_choice": "text"
                    }
                }

            elif choice == "2":
                # User wants Vision LLM processing
                approval_msg = self.get_approval_message(
                    "extract_with_vision",
                    filename=stored_file_name,
                    max_pages=settings.max_pdf_pages
                )

                return {
                    "status": "awaiting_approval",
                    "response": approval_msg,
                    "should_stream": False,
                    "conversation_state": {
                        "step": "vision_approved",
                        "file_path": stored_file_path,
                        "file_name": stored_file_name,
                        "user_choice": "vision"
                    }
                }

            else:  # choice == "3"
                # User wants to save to Notion
                return {
                    "status": "routing",
                    "response": "I'll save this PDF summary to Notion...",
                    "should_stream": False,
                    "next_agent": "notion_agent",
                    "metadata": {
                        "file_path": stored_file_path,
                        "file_name": stored_file_name,
                        "action": "save_pdf"
                    },
                    "conversation_state": {}
                }

        # Step 3: Extract text with pypdf (after approval)
        if workflow_step == "text_extraction_approved":
            if not context.get("approval_granted"):
                # Cleanup file
                if stored_file_path:
                    self.pdf_tools.cleanup_file(stored_file_path)

                return {
                    "status": "complete",
                    "response": "Okay, I won't process the PDF. Let me know if you need anything else!",
                    "should_stream": False,
                    "conversation_state": {}
                }

            # Extract text
            result = await self._extract_text_and_analyze(stored_file_path, stored_file_name, context)

            # Cleanup file after processing
            self.pdf_tools.cleanup_file(stored_file_path)

            return result

        # Step 4: Process with Vision LLM (after approval)
        if workflow_step == "vision_approved":
            if not context.get("approval_granted"):
                # Cleanup file
                if stored_file_path:
                    self.pdf_tools.cleanup_file(stored_file_path)

                return {
                    "status": "complete",
                    "response": "Okay, I won't process the PDF. Let me know if you need anything else!",
                    "should_stream": False,
                    "conversation_state": {}
                }

            # Process with Vision LLM
            result = await self._process_with_vision_and_analyze(
                stored_file_path,
                stored_file_name,
                context
            )

            # Cleanup file after processing
            self.pdf_tools.cleanup_file(stored_file_path)

            return result

        # Unknown state - reset and cleanup
        if stored_file_path:
            self.pdf_tools.cleanup_file(stored_file_path)

        return {
            "status": "error",
            "response": "I'm sorry, I lost track of our conversation. Please send the PDF again.",
            "should_stream": False,
            "conversation_state": {}
        }

    def _initialize_llm(self):
        """
        Initialize the OpenAI LLM for content analysis.

        Returns:
            ChatOpenAI: Configured LLM instance
        """
        logger.info(f"Initializing OpenAI model for PDF analysis: {settings.llm_model}")
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            max_tokens=settings.llm_max_tokens,
        )

    def _format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.

        Args:
            size_bytes: File size in bytes

        Returns:
            Formatted string (e.g., "2.5 MB")
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    async def _extract_text_and_analyze(
        self,
        file_path: str,
        file_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract text from PDF and generate analysis using LLM.

        Args:
            file_path: Path to PDF file
            file_name: Original filename
            context: Conversation context

        Returns:
            Response dictionary with analysis
        """
        logger.info(f"Extracting text from PDF: {file_path}")

        try:
            # Extract text using pypdf
            extraction_result = await self.pdf_tools.extract_text_pypdf(file_path)

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

            # Got text - analyze it with LLM
            text = extraction_result['text']
            page_count = extraction_result['page_count']
            word_count = extraction_result['word_count']

            logger.info(f"Extracted {word_count} words from {page_count} pages")

            # Build analysis prompt
            analysis_prompt = f"""
I've extracted text from a PDF file: {file_name}

**Pages:** {page_count}
**Word Count:** {word_count}

**Content:**
{text[:5000]}{'...' if len(text) > 5000 else ''}

Please provide a concise analysis with:
1. Document type and purpose (2-3 sentences)
2. Key sections or topics covered (bullet points)
3. Main takeaways or conclusions (3-5 points)

Keep your response clear and well-formatted.
"""

            # Generate analysis using LLM
            system_prompt = SystemMessage(content=self.system_prompt)
            user_message = HumanMessage(content=analysis_prompt)

            try:
                response = self.llm.invoke([system_prompt, user_message])
                analysis = response.content

                # Add post-extraction offer
                flow = self.prompt_config["interaction_flow"]["step_7_post_extraction"]
                post_message = flow["message"]

                final_response = f"""✅ **PDF Text Extraction Complete**

**File:** {file_name}
**Pages:** {page_count}
**Word Count:** {word_count}

{analysis}

---

{post_message}
"""

                return {
                    "status": "complete",
                    "response": final_response,
                    "should_stream": True,
                    "metadata": {
                        "file_name": file_name,
                        "page_count": page_count,
                        "word_count": word_count,
                        "extraction_method": "pypdf",
                        "extraction_success": True
                    },
                    "conversation_state": {
                        "step": "offer_notion_save",
                        "file_name": file_name,
                        "content": text
                    }
                }

            except Exception as e:
                logger.error(f"Error analyzing PDF content: {e}", exc_info=True)
                return {
                    "status": "error",
                    "response": f"I extracted the text but encountered an error during analysis: {str(e)}",
                    "should_stream": False,
                    "conversation_state": {}
                }

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}", exc_info=True)
            error_msg = self.get_error_message("extraction_failed", error=str(e))
            return {
                "status": "error",
                "response": error_msg,
                "should_stream": False,
                "conversation_state": {}
            }

    async def _process_with_vision_and_analyze(
        self,
        file_path: str,
        file_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process PDF with Vision LLM and generate analysis.

        Args:
            file_path: Path to PDF file
            file_name: Original filename
            context: Conversation context

        Returns:
            Response dictionary with analysis
        """
        logger.info(f"Processing PDF with Vision LLM: {file_path}")

        try:
            # Process with Vision LLM
            vision_result = await self.pdf_tools.extract_with_vision(
                file_path,
                max_pages=settings.max_pdf_pages
            )

            if not vision_result['success']:
                error_msg = self.get_error_message(
                    "vision_failed",
                    error=vision_result.get('error', 'Unknown error')
                )
                return {
                    "status": "error",
                    "response": error_msg,
                    "should_stream": False,
                    "conversation_state": {}
                }

            # Got content from Vision LLM
            content = vision_result['content']
            pages_processed = vision_result['pages_processed']
            word_count = vision_result['word_count']

            logger.info(f"Vision LLM processed {pages_processed} pages, extracted {word_count} words")

            # The Vision LLM already provides structured analysis
            # Add post-extraction offer
            flow = self.prompt_config["interaction_flow"]["step_7_post_extraction"]
            post_message = flow["message"]

            final_response = f"""✅ **PDF Vision Processing Complete**

**File:** {file_name}
**Pages Processed:** {pages_processed} (max {settings.max_pdf_pages})
**Method:** Vision LLM (gpt-4o)

{content}

---

{post_message}
"""

            return {
                "status": "complete",
                "response": final_response,
                "should_stream": True,
                "metadata": {
                    "file_name": file_name,
                    "pages_processed": pages_processed,
                    "word_count": word_count,
                    "extraction_method": "vision_llm",
                    "extraction_success": True
                },
                "conversation_state": {
                    "step": "offer_notion_save",
                    "file_name": file_name,
                    "content": content
                }
            }

        except Exception as e:
            logger.error(f"Error processing PDF with Vision LLM {file_path}: {e}", exc_info=True)
            error_msg = self.get_error_message("vision_failed", error=str(e))
            return {
                "status": "error",
                "response": error_msg,
                "should_stream": False,
                "conversation_state": {}
            }
