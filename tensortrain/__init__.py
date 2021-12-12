"""Tensor-train methods for many-body physics."""
import logging

from tensortrain.basics import (
    AXES_O,
    AXES_S,
    State,
    chain,
    herm_linear_operator,
    inner,
    Operator,
)

LOGGER = logging.getLogger(__name__)


def setup_logging(level):
    """Set logging level and handler."""
    try:  # use colored log if available
        import colorlog  # pylint: disable=import-outside-toplevel
    except ImportError:  # use standard logging
        logging.basicConfig()
    else:
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s%(reset)s:%(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
