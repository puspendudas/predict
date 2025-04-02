import logging
import os
from datetime import datetime

def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Generate log filename with timestamp
    log_filename = f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # This will also print logs to console
        ]
    )

    # Create logger
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    return logger 