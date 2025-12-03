import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import emoji

class MetricsLogger:
    """Singleton logger for metrics evaluation."""
    
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if MetricsLogger._initialized:
            return
        
        self._loggers = {}
        MetricsLogger._initialized = True

    @classmethod
    def get_instance(cls) -> 'MetricsLogger':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _create_logger(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        console_output: bool = True
    ) -> logging.Logger:
        """Create a configured logger instance."""
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Formatter with emoji support
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

    def get_metrics_logger(
        self,
        log_file: Optional[str] = None,
        level: int = logging.INFO
    ) -> logging.Logger:
        """Get the metrics evaluation logger."""
        
        if 'metrics' not in self._loggers:
            self._loggers['metrics'] = self._create_logger(
                name='metrics_evaluation',
                log_file=log_file,
                level=level
            )
        return self._loggers['metrics']

    def get_execution_logger(
        self,
        log_file: Optional[str] = None,
        level: int = logging.INFO
    ) -> logging.Logger:
        """Get the execution logger."""
        
        if 'execution' not in self._loggers:
            self._loggers['execution'] = self._create_logger(
                name='query_execution',
                log_file=log_file,
                level=level
            )
        return self._loggers['execution']


# Convenience functions for emoji-enhanced logging
def log_with_emoji(logger: logging.Logger, level: int, message: str, emoji_name: str = None):
    """Log message with optional emoji prefix."""
    if emoji_name:
        emoji_char = emoji.emojize(f":{emoji_name}:", language='alias')
        message = f"{emoji_char} {message}"
    logger.log(level, message)


# Singleton instance
_metrics_logger = MetricsLogger.get_instance().get_metrics_logger()