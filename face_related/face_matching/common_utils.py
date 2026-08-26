"""
Common Utilities for Face Recognition Matching
"""
import logging, time, enum
from typing import Optional, NamedTuple


class Colors(str, enum.Enum):
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = PALE = "\033[93m"
    FAIL = RED = "\033[91m"
    ENDC = RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ITALIC = "\033[3m"
    ORANGE = "\033[38;5;208m"
    PURPLE = "\033[38;5;129m"

    def colorize(self, text: str,) -> str:
        """
        Colorize the given text with the color represented by this enum member.

        >>> Colors.GREEN.colorize("This text will be green")
        This text will be green

        >>> Colors.FAIL.colorize("This text will be red")
        This text will be red

        >>> Colors.GREEN.colorize(Colors.BOLD.colorize("This text will be bold and green"))
        This text will be bold and green
        """
        return f"{self.value}{text}{Colors.ENDC.value}"

    # extend Colors class to register new colors and styles
    @classmethod
    def register(cls, name: str, color_code: str):
        """
        Register a new color or style with the given name and color code.

        >>> Colors.register("MAGENTA", "\033[35m")
        >>> Colors.MAGENTA.colorize("This text will be magenta")
        This text will be magenta
        """
        setattr(cls, name, color_code)

class StandardColorizer(NamedTuple):
    """
    A class to represent a colorized string.

    >>> StandardColorizer("This text will be green", Colors.GREEN)
    This text will be green
    """
    red: str = lambda text: Colors.RED.colorize(text)
    green: str = lambda text: Colors.GREEN.colorize(text)
    blue: str = lambda text: Colors.BLUE.colorize(text)
    B: str = lambda text: Colors.BOLD.colorize(text)
    U: str = lambda text: Colors.UNDERLINE.colorize(text)
    I: str = lambda text: Colors.ITALIC.colorize(text)


def log_entry_and_exit(logger, level: str = "info", logging_tag: Optional[str] = None):
    """Decorator to log entry and exit of a function.
    - if the decorated function is a method, it will use the instance's `log_message` method if available, or the instance's `logging_tag` attribute if available.
    Otherwise, it will use the provided `logging_tag` or just log the message without any tag.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            _self = args[0] if args else None   # if the decorated function is a method, _self will be the instance; otherwise None
            if _self and getattr(_self, "log_message", None):
                log_message = _self.log_message
            elif _self and getattr(_self, "logging_tag", None):
                log_message = lambda msg: f"{_self.logging_tag}: {msg}"
            else:
                log_message = lambda msg: f"{logging_tag}: {msg}" if logging_tag else msg
            
            logger.log(getattr(logging, level.upper()), log_message(f"Entering {func.__qualname__}"))
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            logger.log(getattr(logging, level.upper()), log_message(f"Exiting {func.__qualname__}"))
            logger.info(log_message(f"Execution time: {end_time - start_time:.6f} seconds"))
            return result
        return wrapper
    return decorator
