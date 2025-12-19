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

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(base_handler)
    root_logger.addHandler(error_handler)


    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     handlers=[base_handler, console_handler, error_handler]
    # )