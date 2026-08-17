"""
Structured Logging utility for Production RAG Engine.
Provides clean terminal and file logging with Rich integration.
"""
import logging
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure global logging format and level.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = []
    if RICH_AVAILABLE:
        handlers.append(
            RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_path=False,
            )
        )
    else:
        standard_handler = logging.StreamHandler(sys.stdout)
        standard_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        handlers.append(standard_handler)

    # Root logger configuration
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy_logger in ["httpx", "chromadb", "urllib3", "posthog", "onnxruntime"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance with the given module name.
    """
    return logging.getLogger(name or "rag_engine")
