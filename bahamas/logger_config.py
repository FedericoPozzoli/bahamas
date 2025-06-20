# logger_config.py
import logging

# Configure logger
logger = logging.getLogger('BAHAMAS')
logger.setLevel(logging.DEBUG)

# Avoid adding multiple handlers if already added
if not logger.handlers:
    # Add a console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
