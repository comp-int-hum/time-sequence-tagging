import logging
import sys

def setup_logging():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    base_handler = logging.FileHandler("sconstruct.log")
    base_handler.setLevel(logging.INFO)

    error_handler = logging.FileHandler("sconstruct_debug.log")
    error_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    for handler in [console_handler, base_handler, error_handler]:
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[base_handler, console_handler, error_handler]
    )