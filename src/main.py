"""
Main FastAPI application module.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.health import router as health_router
from src.api.telegram import router as telegram_router
from src.api.models import ErrorResponse
from src.core.config import get_config, validate_config_completeness
from src.core.database import DatabaseManager, get_database_manager
from src.core.logging import get_logger, setup_logging
from src.core.middleware import (
    ErrorHandlingMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)

# Setup logging first
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting application...")

    try:
        # Load and validate configuration
        config = get_config()
        validate_config_completeness(config)
        app.state.config = config

        logger.info(
            f"Configuration loaded: {config.app_name} (Environment: {config.debug and 'Development' or 'Production'})"
        )

        # Initialize database
        db_manager = await get_database_manager()
        await db_manager.initialize()
        await db_manager.create_tables()
        app.state.db_manager = db_manager

        logger.info("Database initialized and tables created successfully")

        # Log startup summary
        logger.info(
            f"Application started successfully | "
            f"Host: {config.host} | Port: {config.port} | "
            f"Debug: {config.debug}"
        )

        yield

    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise

    finally:
        # Shutdown
        logger.info("Shutting down application...")

        # Close database connections
        if hasattr(app.state, "db_manager"):
            await app.state.db_manager.close()
            logger.info("Database connections closed")

        logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    # Load configuration
    config = get_config()

    # Setup logging
    log_file = Path("logs") / "app.log" if not config.debug else None
    setup_logging(
        log_level=config.log_level, log_file=log_file, app_name=config.app_name
    )

    # Create FastAPI instance
    app = FastAPI(
        title=config.app_name,
        description="AI-powered Telegram bot with FastAPI backend",
        version="1.0.0",
        debug=config.debug,
        lifespan=lifespan,
        docs_url="/docs" if config.debug else None,  # Disable docs in production
        redoc_url="/redoc" if config.debug else None,  # Disable redoc in production
        openapi_url=(
            "/openapi.json" if config.debug else None
        ),  # Disable OpenAPI in production
    )

    # Store config in app state for access in routes
    app.state.config = config

    # Add middleware in correct order (outermost first)

    # Trusted Host Middleware (security)
    if not config.debug:
        # In production, restrict to specific hosts
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=["*"]  # Configure based on your domain
        )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.debug else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        max_age=3600,
    )

    # GZip Middleware for compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)

    # Rate limiting (optional, can be configured)
    if not config.debug:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

    # Include routers
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(telegram_router)

    # Exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        request_id = getattr(request.state, "request_id", None)

        logger.warning(
            f"HTTP exception | Request ID: {request_id} | "
            f"Status: {exc.status_code} | Detail: {exc.detail}"
        )

        error_response = ErrorResponse(
            error=f"HTTP {exc.status_code}", detail=exc.detail, request_id=request_id
        )

        return JSONResponse(
            status_code=exc.status_code, content=error_response.model_dump()
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle validation errors."""
        request_id = getattr(request.state, "request_id", None)

        logger.warning(
            f"Validation error | Request ID: {request_id} | " f"Errors: {exc.errors()}"
        )

        # Format validation errors
        errors = []
        for error in exc.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            errors.append(f"{field}: {error['msg']}")

        error_response = ErrorResponse(
            error="Validation Error", detail="; ".join(errors), request_id=request_id
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions."""
        request_id = getattr(request.state, "request_id", None)

        logger.error(
            f"Uncaught exception | Request ID: {request_id} | " f"Error: {str(exc)}",
            exc_info=True,
        )

        error_response = ErrorResponse(
            error="Internal Server Error",
            detail=str(exc) if config.debug else "An unexpected error occurred",
            request_id=request_id,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(),
        )

    # Root endpoint
    @app.get("/", tags=["root"])
    @app.options(
        "/", summary="CORS Preflight for Health Check", include_in_schema=False
    )  # Hide from OpenAPI docs
    async def root():
        """Root endpoint with basic application info."""
        return {
            "name": config.app_name,
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs" if config.debug else None,
            "health": "/api/v1/health",
        }

    # Favicon handler to avoid 404s
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """Return empty favicon to avoid 404 errors."""
        return JSONResponse(content={}, status_code=204)

    return app


# Create the FastAPI app instance
app = create_app()


if __name__ == "__main__":
    """
    Run the application using uvicorn when executed directly.
    """
    import uvicorn

    config = get_config()

    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s | %(levelname)s | %(message)s"
    log_config["formatters"]["access"]["fmt"] = (
        "%(asctime)s | %(levelname)s | %(client_addr)s | "
        "%(request_line)s | %(status_code)s"
    )

    # Run the server
    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower(),
        log_config=log_config,
        access_log=True,
        # SSL configuration for production
        ssl_keyfile=None,  # Add path to SSL key file for HTTPS
        ssl_certfile=None,  # Add path to SSL certificate file for HTTPS
    )
