"""
NotionTools - Tools for interacting with Notion API

This module provides tools for creating and managing Notion pages:
- Creating simple pages with title and content
- Formatting content with markdown
- Managing page metadata
"""

import logging
from typing import Dict, Any, Optional
from notion_client import Client as NotionClient
from notion_client.errors import APIResponseError

from src.core.config import get_config

logger = logging.getLogger(__name__)
settings = get_config()


class NotionTools:
    """
    Tools for interacting with Notion API.

    Provides methods for:
    - Creating simple Notion pages
    - Formatting content for Notion
    - Managing page metadata

    Example usage:
        tools = NotionTools()

        # Create a page
        result = await tools.create_page(
            title="My Article",
            content="This is the content...",
            metadata={"url": "https://example.com"}
        )
        print(f"Page created: {result['page_url']}")
    """

    def __init__(self, api_key: Optional[str] = None, parent_page_id: Optional[str] = None):
        """
        Initialize Notion tools with API credentials.

        Args:
            api_key: Optional Notion API key. If not provided, will use
                    settings.notion_api_key from configuration.
            parent_page_id: Optional parent page ID. If not provided, will use
                           settings.notion_default_parent_page_id from configuration.

        Raises:
            ValueError: If no API key or parent page ID is provided and none exists in config
        """
        self.api_key = api_key or settings.notion_api_key
        self.parent_page_id = parent_page_id or settings.notion_default_parent_page_id

        if not self.api_key:
            raise ValueError(
                "Notion API key not configured. "
                "Please set NOTION_API_KEY in your .env.dev file. "
                "Get your API key from https://www.notion.so/my-integrations"
            )

        if not self.parent_page_id:
            raise ValueError(
                "Notion parent page ID not configured. "
                "Please set NOTION_DEFAULT_PARENT_PAGE_ID in your .env.dev file."
            )

        self.client = NotionClient(auth=self.api_key)
        logger.info("NotionTools initialized successfully")

    async def create_page(
        self,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new page in Notion.

        This method creates a simple page with a title and content. Content is
        formatted as markdown-style blocks. Metadata (like URLs or filenames)
        is included at the top of the page.

        Args:
            title: Page title
            content: Page content (supports basic markdown)
            metadata: Optional metadata to include (e.g., {"url": "...", "source": "..."})

        Returns:
            Dictionary containing:
                - success (bool): Whether creation was successful
                - page_id (str): Notion page ID
                - page_url (str): Public URL to the page
                - title (str): Page title
                - error (str): Error message if creation failed

        Example:
            result = await tools.create_page(
                title="Article Title",
                content="This is the article content...",
                metadata={"url": "https://example.com", "source": "web"}
            )
        """
        if not title or not title.strip():
            return {
                'success': False,
                'page_id': '',
                'page_url': '',
                'title': '',
                'error': 'Page title is required'
            }

        if not content or not content.strip():
            return {
                'success': False,
                'page_id': '',
                'page_url': '',
                'title': title,
                'error': 'Page content is required'
            }

        logger.info(f"Creating Notion page: {title}")

        try:
            # Build page blocks (Notion's content format)
            blocks = []

            # Add metadata section if provided
            if metadata:
                metadata_lines = []
                for key, value in metadata.items():
                    if value:
                        metadata_lines.append(f"**{key.title()}:** {value}")

                if metadata_lines:
                    # Add metadata as a callout block
                    blocks.append({
                        "object": "block",
                        "type": "callout",
                        "callout": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": "\n".join(metadata_lines)}
                            }],
                            "icon": {"emoji": "📌"}
                        }
                    })

                    # Add divider
                    blocks.append({
                        "object": "block",
                        "type": "divider",
                        "divider": {}
                    })

            # Add main content as paragraphs
            # Split content into paragraphs (by double newlines)
            paragraphs = content.split('\n\n')

            for paragraph in paragraphs:
                if paragraph.strip():
                    # Split paragraph by single newlines to handle bullet lists
                    lines = paragraph.split('\n')

                    for line in lines:
                        line_text = line.strip()
                        if not line_text:
                            continue

                        # Check if it's a heading (starts with #)
                        if line_text.startswith('###'):
                            blocks.append({
                                "object": "block",
                                "type": "heading_3",
                                "heading_3": {
                                    "rich_text": [{
                                        "type": "text",
                                        "text": {"content": line_text.replace('###', '').strip()}
                                    }]
                                }
                            })
                        elif line_text.startswith('##'):
                            blocks.append({
                                "object": "block",
                                "type": "heading_2",
                                "heading_2": {
                                    "rich_text": [{
                                        "type": "text",
                                        "text": {"content": line_text.replace('##', '').strip()}
                                    }]
                                }
                            })
                        elif line_text.startswith('#'):
                            blocks.append({
                                "object": "block",
                                "type": "heading_1",
                                "heading_1": {
                                    "rich_text": [{
                                        "type": "text",
                                        "text": {"content": line_text.replace('#', '').strip()}
                                    }]
                                }
                            })
                        # Check if it's a bullet list item
                        elif line_text.startswith('- ') or line_text.startswith('* '):
                            blocks.append({
                                "object": "block",
                                "type": "bulleted_list_item",
                                "bulleted_list_item": {
                                    "rich_text": [{
                                        "type": "text",
                                        "text": {"content": line_text[2:].strip()}
                                    }]
                                }
                            })
                        else:
                            # Regular paragraph
                            # Split long text into chunks (Notion has 2000 char limit per block)
                            chunk_size = 2000
                            for i in range(0, len(line_text), chunk_size):
                                chunk = line_text[i:i+chunk_size]
                                blocks.append({
                                    "object": "block",
                                    "type": "paragraph",
                                    "paragraph": {
                                        "rich_text": [{
                                            "type": "text",
                                            "text": {"content": chunk}
                                        }]
                                    }
                                })

            # Create the page
            new_page = self.client.pages.create(
                parent={"page_id": self.parent_page_id},
                properties={
                    "title": {
                        "title": [{
                            "text": {"content": title}
                        }]
                    }
                },
                children=blocks
            )

            page_id = new_page['id']
            page_url = new_page['url']

            logger.info(f"Successfully created Notion page: {page_id}")

            return {
                'success': True,
                'page_id': page_id,
                'page_url': page_url,
                'title': title,
            }

        except APIResponseError as e:
            error_msg = f"Notion API error: {str(e)}"
            logger.error(f"Failed to create Notion page: {error_msg}")
            return {
                'success': False,
                'page_id': '',
                'page_url': '',
                'title': title,
                'error': error_msg
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to create Notion page: {error_msg}", exc_info=True)
            return {
                'success': False,
                'page_id': '',
                'page_url': '',
                'title': title,
                'error': error_msg
            }
