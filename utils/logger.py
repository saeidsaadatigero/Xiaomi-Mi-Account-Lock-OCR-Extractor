"""unlock_code_extractor/utils/logger.py — Structlog configuration for structured logging."""

import logging
import sys

import structlog


def setup_logging(verbose: bool = False) -> structlog.BoundLogger:
    """
    Configure and return a structlog logger instance.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.

    Returns:
        Configured structlog BoundLogger.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()
