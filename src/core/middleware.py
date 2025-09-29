"""
Custom middleware for the FastAPI application.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.models import ErrorResponse
from src.core.logging import get_logger


logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all incoming requests and responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            f"Request started | ID: {request_id} | "
            f"Method: {request.method} | Path: {request.url.path} | "
            f"Client: {request.client.host if request.client else 'Unknown'}"
        )
        
        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        # Calculate request duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(
            f"Request completed | ID: {request_id} | "
            f"Status: {response.status_code} | "
            f"Duration: {duration_ms:.2f}ms"
        )
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware to catch unhandled exceptions.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except Exception as exc:
            # Get request ID if available
            request_id = getattr(request.state, 'request_id', None)
            
            # Log the error with full traceback
            logger.error(
                f"Unhandled exception | Request ID: {request_id} | "
                f"Path: {request.url.path} | Error: {str(exc)}",
                exc_info=True
            )
            
            # Create error response
            error_response = ErrorResponse(
                error="Internal Server Error",
                detail=str(exc) if request.app.state.config.debug else None,
                request_id=request_id
            )
            
            # Return JSON error response
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.model_dump()
            )


class CORSMiddleware:
    """
    Custom CORS middleware with more control than the default FastAPI CORS middleware.
    """
    
    def __init__(
        self,
        app,
        allow_origins: list = ["*"],
        allow_credentials: bool = False,
        allow_methods: list = ["*"],
        allow_headers: list = ["*"],
        max_age: int = 600
    ):
        self.app = app
        self.allow_origins = allow_origins
        self.allow_credentials = allow_credentials
        self.allow_methods = allow_methods
        self.allow_headers = allow_headers
        self.max_age = max_age
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        # Handle preflight requests
        if request.method == "OPTIONS":
            return self._preflight_response(request)
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        origin = request.headers.get("origin")
        
        if origin:
            # Check if origin is allowed
            if self._is_origin_allowed(origin):
                response.headers["Access-Control-Allow-Origin"] = origin
                
                if self.allow_credentials:
                    response.headers["Access-Control-Allow-Credentials"] = "true"
                
                # Add Vary header for caching
                vary_headers = response.headers.get("Vary", "")
                if vary_headers:
                    vary_headers += ", Origin"
                else:
                    vary_headers = "Origin"
                response.headers["Vary"] = vary_headers
        
        return response
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if the origin is allowed."""
        if "*" in self.allow_origins:
            return True
        return origin in self.allow_origins
    
    def _preflight_response(self, request: Request) -> Response:
        """Handle preflight OPTIONS requests."""
        response = Response(content="", status_code=200)
        
        origin = request.headers.get("origin")
        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
            
            # Handle requested methods
            requested_method = request.headers.get("Access-Control-Request-Method")
            if requested_method:
                if "*" in self.allow_methods or requested_method in self.allow_methods:
                    response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            
            # Handle requested headers
            requested_headers = request.headers.get("Access-Control-Request-Headers")
            if requested_headers:
                if "*" in self.allow_headers:
                    response.headers["Access-Control-Allow-Headers"] = requested_headers
                else:
                    response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
            
            # Set max age for preflight caching
            response.headers["Access-Control-Max-Age"] = str(self.max_age)
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware.
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}  # IP -> (timestamp, count)
        self.window = 60  # 1 minute window
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        current_time = time.time()
        
        # Check if this is a health check endpoint (exempt from rate limiting)
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Clean old entries
        self._clean_old_entries(current_time)
        
        # Check rate limit
        if client_ip in self.request_counts:
            timestamp, count = self.request_counts[client_ip]
            
            if current_time - timestamp < self.window:
                if count >= self.requests_per_minute:
                    logger.warning(f"Rate limit exceeded for {client_ip}")
                    
                    error_response = ErrorResponse(
                        error="Rate limit exceeded",
                        detail=f"Maximum {self.requests_per_minute} requests per minute"
                    )
                    
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content=error_response.model_dump(),
                        headers={"Retry-After": str(self.window)}
                    )
                
                # Increment count
                self.request_counts[client_ip] = (timestamp, count + 1)
            else:
                # Reset counter for new window
                self.request_counts[client_ip] = (current_time, 1)
        else:
            # First request from this IP
            self.request_counts[client_ip] = (current_time, 1)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        if client_ip in self.request_counts:
            _, count = self.request_counts[client_ip]
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(max(0, self.requests_per_minute - count))
            response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window))
        
        return response
    
    def _clean_old_entries(self, current_time: float):
        """Remove old entries from the request counts."""
        to_remove = []
        for ip, (timestamp, _) in self.request_counts.items():
            if current_time - timestamp > self.window:
                to_remove.append(ip)
        
        for ip in to_remove:
            del self.request_counts[ip]


class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to compress responses when appropriate.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response
        
        # Only compress certain content types and sizes
        content_type = response.headers.get("content-type", "")
        compressible_types = ["application/json", "text/", "application/javascript"]
        
        should_compress = any(ct in content_type for ct in compressible_types)
        
        if should_compress and hasattr(response, "body"):
            # Note: Actual compression would require more complex handling
            # This is a placeholder for the compression logic
            response.headers["Content-Encoding"] = "gzip"
        
        return response
