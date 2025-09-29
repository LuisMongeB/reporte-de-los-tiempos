"""
Health check endpoints for the Telegram AI Agent API.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.models import (
    ConfigSummary,
    DatabaseStatus,
    DetailedHealthResponse,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
)
from src.core.config import BaseConfig, get_config
from src.core.database import DatabaseManager, get_database_manager

# Setup logging
logger = logging.getLogger(__name__)

# Create router with health prefix and tags
router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


# Dependency functions for dependency injection
async def get_config_dependency() -> BaseConfig:
    """
    Get configuration dependency for health endpoints.

    Returns:
        BaseConfig: Application configuration

    Raises:
        HTTPException: If configuration cannot be loaded
    """
    try:
        return get_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration unavailable",
        )


async def get_database_manager_dependency() -> DatabaseManager:
    """
    Get database manager dependency for health endpoints.

    Returns:
        DatabaseManager: Database manager instance

    Raises:
        HTTPException: If database manager cannot be initialized
    """
    try:
        return await get_database_manager()
    except Exception as e:
        logger.error(f"Failed to get database manager: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database manager unavailable",
        )


# Health endpoints


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Basic Health Check",
    description="Simple health check endpoint that returns service status and timestamp",
)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.

    Returns a simple health status with timestamp. This endpoint should be
    fast and lightweight, suitable for load balancer health checks.

    Returns:
        HealthResponse: Basic health status and timestamp
    """
    start_time = time.time()

    try:
        # Create basic health response
        response = HealthResponse(health_status=HealthStatus.HEALTHY)

        # Log response time for monitoring
        response_time_ms = (time.time() - start_time) * 1000
        logger.debug(f"Health check completed in {response_time_ms:.2f}ms")

        # Warn if response time is unexpectedly high
        if response_time_ms > 100:
            logger.warning(
                f"Health check took {response_time_ms:.2f}ms (> 100ms threshold)"
            )

        return response

    except Exception as e:
        logger.error(f"Health check failed: {e}")

        # Return unhealthy status but don't raise exception
        # Health endpoints should always return a response if possible
        return HealthResponse(health_status=HealthStatus.UNHEALTHY)


@router.get(
    "/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed Health Check",
    description="Comprehensive health check with database status, configuration summary, and performance metrics",
)
async def detailed_health_check(
    config: Annotated[BaseConfig, Depends(get_config_dependency)],
    db_manager: Annotated[DatabaseManager, Depends(get_database_manager_dependency)],
) -> DetailedHealthResponse:
    """
    Detailed health check endpoint with comprehensive system information.

    This endpoint provides detailed information about:
    - Database connectivity and performance
    - Configuration summary (without sensitive data)
    - Service uptime and version info

    Args:
        config: Application configuration (injected)
        db_manager: Database manager instance (injected)

    Returns:
        DetailedHealthResponse: Comprehensive health information
    """
    start_time = time.time()

    try:
        # Check database connectivity and measure response time
        db_start_time = time.time()

        try:
            is_db_connected = await db_manager.check_connection()
            db_response_time_ms = (time.time() - db_start_time) * 1000

            database_status = DatabaseStatus(
                is_connected=is_db_connected,
                response_time_ms=db_response_time_ms,
                error_message=None,
            )

            if is_db_connected:
                logger.debug(
                    f"Database connection check passed in {db_response_time_ms:.2f}ms"
                )
            else:
                logger.warning("Database connection check failed")

        except Exception as db_error:
            db_response_time_ms = (time.time() - db_start_time) * 1000
            logger.error(f"Database connection check error: {db_error}")

            database_status = DatabaseStatus(
                is_connected=False,
                response_time_ms=db_response_time_ms,
                error_message=str(db_error),
            )

        # Get safe configuration summary
        try:
            config_summary = ConfigSummary.from_config(config)
            logger.debug("Configuration summary generated successfully")
        except Exception as config_error:
            logger.error(f"Failed to generate config summary: {config_error}")
            # Provide a minimal config summary as fallback
            config_summary = ConfigSummary(
                app_name=getattr(config, "app_name", "Unknown"),
                debug=getattr(config, "debug", False),
                log_level=getattr(config, "log_level", "INFO"),
                database_type="unknown",
                telegram_configured=False,
                openai_configured=False,
                webhook_configured=False,
                temp_dir="unknown",
                max_file_size_mb=0,
            )

        # Calculate total endpoint execution time
        total_response_time = time.time() - start_time

        # Create detailed health response
        detailed_response = DetailedHealthResponse(
            database=database_status,
            config=config_summary,
            uptime_seconds=total_response_time,  # Placeholder - could track actual uptime
            version=None,  # Could be populated from pyproject.toml or environment
        )

        logger.info(
            f"Detailed health check completed in {total_response_time * 1000:.2f}ms"
        )

        return detailed_response

    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")

        # Return an error response with what information we can gather
        error_response = DetailedHealthResponse(
            health_status=HealthStatus.UNHEALTHY,
            database=DatabaseStatus(
                is_connected=False, error_message=f"Health check failed: {str(e)}"
            ),
            config=ConfigSummary(
                app_name="Unknown",
                debug=False,
                log_level="ERROR",
                database_type="unknown",
                telegram_configured=False,
                openai_configured=False,
                webhook_configured=False,
                temp_dir="unknown",
                max_file_size_mb=0,
            ),
        )

        return error_response
