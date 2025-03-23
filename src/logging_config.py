# src/logging_config.py
import logging
import os

def configure_logging(log_directory='logs'):
    # Create the logs directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)

    # Configure the root logger (will log from all modules)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create loggers for specific modules
    modules = ['train', 'predict', 'api']
    loggers = {}

    for module in modules:
        # Set up logging for each module
        logger = logging.getLogger(f'ml_app.{module}')
        
        # Create a file handler for each module's logs
        file_handler = logging.FileHandler(f'{log_directory}/{module}.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add the file handler to the logger
        logger.addHandler(file_handler)
        
        loggers[module] = logger

    return loggers
