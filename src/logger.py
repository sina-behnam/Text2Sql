import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, Any
from enum import Enum
import threading
from functools import wraps

# Optional imports for enhanced logging
try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class LogLevel(Enum):
    """Enumeration for log levels"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ComponentType(Enum):
    """Enumeration for different system components"""
    DATALOADER = "dataloader"
    PIPELINE = "pipeline"
    MODEL = "model"
    EVALUATOR = "evaluator"
    UTILS = "utils"
    MAIN = "main"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = HAS_COLORAMA and sys.stderr.isatty()
        
        if self.use_colors:
            self.colors = {
                'DEBUG': Fore.CYAN,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
            }
            self.reset = Style.RESET_ALL
        else:
            self.colors = {}
            self.reset = ""
    
    def format(self, record):
        if self.use_colors and record.levelname in self.colors:
            # Color the level name
            record.levelname = f"{self.colors[record.levelname]}{record.levelname}{self.reset}"
            
            # Color the message based on level
            if record.levelno >= logging.ERROR:
                record.msg = f"{self.colors['ERROR']}{record.msg}{self.reset}"
            elif record.levelno >= logging.WARNING:
                record.msg = f"{self.colors['WARNING']}{record.msg}{self.reset}"
        
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'component': getattr(record, 'component', 'unknown'),
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
            
        return json.dumps(log_entry)


class LoggerConfig:
    """Configuration class for the logging system"""
    
    def __init__(
        self,
        log_dir: str = "logs",
        console_level: LogLevel = LogLevel.INFO,
        file_level: LogLevel = LogLevel.DEBUG,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        log_format: str = None,
        use_colors: bool = True,
        use_structured_logging: bool = False,
        component_levels: Dict[ComponentType, LogLevel] = None
    ):
        self.log_dir = Path(log_dir)
        self.console_level = console_level
        self.file_level = file_level
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.use_colors = use_colors and HAS_COLORAMA
        self.use_structured_logging = use_structured_logging
        self.component_levels = component_levels or {}
        
        # Default log format
        if log_format is None:
            self.log_format = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "[%(component)s] - %(funcName)s:%(lineno)d - %(message)s"
            )
        else:
            self.log_format = log_format
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)


class Text2SQLLogger:
    """Centralized logging system for Text2SQL pipeline"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one logger instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: LoggerConfig = None):
        """Initialize the logging system"""
        if hasattr(self, '_initialized'):
            return
            
        self.config = config or LoggerConfig()
        self.loggers = {}
        self._setup_root_logger()
        self._initialized = True
    
    def _setup_root_logger(self):
        """Set up the root logger configuration"""
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.config.console_level.value)
        
        if self.config.use_colors:
            console_formatter = ColoredFormatter(self.config.log_format)
        else:
            console_formatter = logging.Formatter(self.config.log_format)
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Create file handler with rotation
        log_file = self.config.log_dir / "text2sql.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count
        )
        file_handler.setLevel(self.config.file_level.value)
        
        if self.config.use_structured_logging:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(self.config.log_format)
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    def get_logger(
        self, 
        name: str, 
        component: ComponentType = ComponentType.MAIN,
        extra_data: Dict[str, Any] = None
    ) -> logging.Logger:
        """
        Get a logger for a specific component
        
        Args:
            name: Logger name (usually __name__)
            component: Type of component
            extra_data: Additional data to include in log records
            
        Returns:
            Configured logger instance
        """
        logger_key = f"{component.value}.{name}"
        
        if logger_key not in self.loggers:
            logger = logging.getLogger(logger_key)
            
            # Set component-specific log level if configured
            if component in self.config.component_levels:
                logger.setLevel(self.config.component_levels[component].value)
            
            # Create custom adapter to add component info
            logger = ComponentLoggerAdapter(logger, component, extra_data)
            self.loggers[logger_key] = logger
        
        return self.loggers[logger_key]
    
    def log_experiment_start(self, experiment_name: str, config: Dict[str, Any]):
        """Log the start of an experiment with configuration"""
        logger = self.get_logger("experiment", ComponentType.MAIN)
        logger.info(
            f"Starting experiment: {experiment_name}",
            extra={'experiment_config': config}
        )
    
    def log_experiment_end(self, experiment_name: str, results: Dict[str, Any]):
        """Log the end of an experiment with results"""
        logger = self.get_logger("experiment", ComponentType.MAIN)
        logger.info(
            f"Completed experiment: {experiment_name}",
            extra={'experiment_results': results}
        )
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Log model performance metrics"""
        logger = self.get_logger("performance", ComponentType.MODEL)
        logger.info(
            f"Model performance - {model_name}",
            extra={'model_metrics': metrics}
        )
    
    def create_progress_logger(self, total: int, desc: str = "Processing") -> 'ProgressLogger':
        """Create a progress logger that integrates with tqdm if available"""
        return ProgressLogger(self, total, desc)


class ComponentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds component information to log records"""
    
    def __init__(self, logger: logging.Logger, component: ComponentType, extra_data: Dict[str, Any] = None):
        self.component = component
        self.base_extra = extra_data or {}
        super().__init__(logger, {})
    
    def process(self, msg, kwargs):
        """Add component information to log records"""
        extra = kwargs.get('extra', {})
        extra.update(self.base_extra)
        extra['component'] = self.component.value
        kwargs['extra'] = extra
        return msg, kwargs


class ProgressLogger:
    """Progress logger that can integrate with tqdm or provide simple logging"""
    
    def __init__(self, logger_system: Text2SQLLogger, total: int, desc: str = "Processing"):
        self.logger_system = logger_system
        self.total = total
        self.desc = desc
        self.current = 0
        self.logger = logger_system.get_logger("progress", ComponentType.MAIN)
        
        if HAS_TQDM:
            self.progress_bar = tqdm(total=total, desc=desc)
        else:
            self.progress_bar = None
            self.logger.info(f"Starting {desc}: 0/{total}")
    
    def update(self, n: int = 1, **kwargs):
        """Update progress"""
        self.current += n
        
        if self.progress_bar:
            self.progress_bar.update(n)
            if kwargs:
                self.progress_bar.set_postfix(**kwargs)
        else:
            # Log every 10% or so
            if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
                percentage = (self.current / self.total) * 100
                self.logger.info(f"{self.desc}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def close(self):
        """Close the progress logger"""
        if self.progress_bar:
            self.progress_bar.close()
        else:
            self.logger.info(f"Completed {self.desc}: {self.total}/{self.total}")


def setup_logging(config: LoggerConfig = None) -> Text2SQLLogger:
    """
    Set up the centralized logging system
    
    Args:
        config: Logger configuration (optional)
        
    Returns:
        Configured logger system
    """
    return Text2SQLLogger(config)


def get_logger(name: str, component: ComponentType = ComponentType.MAIN) -> logging.Logger:
    """
    Convenience function to get a logger
    
    Args:
        name: Logger name (usually __name__)
        component: Component type
        
    Returns:
        Configured logger
    """
    logger_system = Text2SQLLogger()
    return logger_system.get_logger(name, component)


def log_execution_time(component: ComponentType = ComponentType.MAIN):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__name__, component)
            start_time = datetime.now()
            
            try:
                logger.debug(f"Starting execution of {func.__name__}")
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Completed {func.__name__} in {execution_time:.2f}s",
                    extra={'execution_time': execution_time}
                )
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Failed {func.__name__} after {execution_time:.2f}s: {str(e)}",
                    extra={'execution_time': execution_time, 'error': str(e)}
                )
                raise
                
        return wrapper
    return decorator


# Example configuration presets
def get_development_config() -> LoggerConfig:
    """Get configuration suitable for development"""
    return LoggerConfig(
        console_level=LogLevel.DEBUG,
        file_level=LogLevel.DEBUG,
        use_colors=True,
        use_structured_logging=False,
        component_levels={
            ComponentType.MODEL: LogLevel.INFO,
            ComponentType.DATALOADER: LogLevel.DEBUG,
        }
    )


def get_production_config() -> LoggerConfig:
    """Get configuration suitable for production"""
    return LoggerConfig(
        console_level=LogLevel.INFO,
        file_level=LogLevel.DEBUG,
        use_colors=False,
        use_structured_logging=True,
        max_file_size=50 * 1024 * 1024,  # 50MB
        backup_count=10,
        component_levels={
            ComponentType.MODEL: LogLevel.WARNING,
            ComponentType.DATALOADER: LogLevel.INFO,
        }
    )


def get_experiment_config() -> LoggerConfig:
    """Get configuration suitable for experiments"""
    return LoggerConfig(
        log_dir="experiment_logs",
        console_level=LogLevel.INFO,
        file_level=LogLevel.DEBUG,
        use_colors=True,
        use_structured_logging=True,
        component_levels={
            ComponentType.PIPELINE: LogLevel.INFO,
            ComponentType.MODEL: LogLevel.INFO,
            ComponentType.EVALUATOR: LogLevel.DEBUG,
        }
    )