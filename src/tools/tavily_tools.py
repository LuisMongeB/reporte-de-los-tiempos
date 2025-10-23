"""
TavilyTools - Tools for extracting and searching content using Tavily API

This module provides tools for interacting with Tavily's Extract and Search APIs.
The Extract API provides clean, AI-optimized content extraction from URLs.
"""

import logging
from typing import Dict, Any, Optional
from tavily import TavilyClient

from src.core.config import get_config

logger = logging.getLogger(__name__)
settings = get_config()


class TavilyTools:
    """
    Tools for interacting with Tavily API.

    Provides methods for:
    - Extracting clean content from URLs using Tavily Extract API
    - Searching the web using Tavily Search API (future)

    Example usage:
        tools = TavilyTools()
        result = await tools.extract_url("https://example.com/article")
        print(f"Title: {result['title']}")
        print(f"Content: {result['content']}")
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily tools with API key.

        Args:
            api_key: Optional Tavily API key. If not provided, will use
                    settings.tavily_api_key from configuration.

        Raises:
            ValueError: If no API key is provided and none exists in config
        """
        self.api_key = api_key or settings.tavily_api_key

        if not self.api_key:
            raise ValueError(
                "Tavily API key not configured. "
                "Please set TAVILY_API_KEY in your .env.dev file. "
                "Get your API key from https://tavily.com"
            )

        self.client = TavilyClient(api_key=self.api_key)
        logger.info("TavilyTools initialized successfully")

    async def extract_url(self, url: str) -> Dict[str, Any]:
        """
        Extract clean content from a URL using Tavily Extract API.

        This method fetches and extracts the main content from a web page,
        removing ads, navigation, and other clutter. Perfect for getting
        AI-ready content for analysis.

        Args:
            url: The URL to extract content from. Must be a valid HTTP/HTTPS URL.

        Returns:
            Dictionary containing:
                - success (bool): Whether extraction was successful
                - url (str): The original URL
                - title (str): Page title (if available)
                - content (str): Extracted clean content
                - raw_content (str): Raw HTML content (if available)
                - word_count (int): Approximate word count
                - error (str): Error message if extraction failed

        Raises:
            ValueError: If URL is invalid or empty
            Exception: If Tavily API call fails

        Example:
            result = await tools.extract_url("https://example.com")
            if result['success']:
                print(f"Extracted {result['word_count']} words")
                print(result['content'][:200])
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")

        url = url.strip()

        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            raise ValueError(
                f"Invalid URL format: {url}. URL must start with http:// or https://"
            )

        logger.info(f"Extracting content from URL: {url}")

        try:
            # Call Tavily Extract API
            # Note: extract() is synchronous in tavily-python
            response = self.client.extract(urls=[url])

            # Tavily returns a list of results, one per URL
            if not response or not response.get('results'):
                logger.warning(f"No content extracted from URL: {url}")
                return {
                    'success': False,
                    'url': url,
                    'title': '',
                    'content': '',
                    'raw_content': '',
                    'word_count': 0,
                    'error': 'No content could be extracted from this URL'
                }

            # Get the first result (we only sent one URL)
            result = response['results'][0]

            # Extract fields
            content = result.get('raw_content', '')
            title = result.get('title', '')

            # Calculate word count
            word_count = len(content.split()) if content else 0

            logger.info(f"Successfully extracted {word_count} words from {url}")

            return {
                'success': True,
                'url': url,
                'title': title,
                'content': content,
                'raw_content': content,  # Tavily provides clean content
                'word_count': word_count,
            }

        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {str(e)}")
            return {
                'success': False,
                'url': url,
                'title': '',
                'content': '',
                'raw_content': '',
                'word_count': 0,
                'error': str(e)
            }

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search the web using Tavily Search API.

        Note: This is a placeholder for future implementation.
        Will be implemented when search functionality is needed.

        Args:
            query: Search query string
            max_results: Maximum number of results (default from config)
            search_depth: "basic" or "advanced" (default from config)

        Returns:
            Dictionary containing search results
        """
        max_results = max_results or settings.tavily_max_results
        search_depth = search_depth or settings.tavily_search_depth

        logger.info(f"Searching Tavily: {query} (depth={search_depth}, max={max_results})")

        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth
            )

            return {
                'success': True,
                'query': query,
                'results': response.get('results', []),
                'answer': response.get('answer', ''),
            }

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            return {
                'success': False,
                'query': query,
                'results': [],
                'error': str(e)
            }
