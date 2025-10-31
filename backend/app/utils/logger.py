import logging
from app.utils.config import settings

def setup_logger(name: str = None) -> logging.Logger:
    name = name or settings.APP_NAME
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(ch)
    return logger
