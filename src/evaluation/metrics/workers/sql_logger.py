import logging
from pathlib import Path
from typing import Optional


class SQLLogger:
    _instance: Optional["SQLLogger"] = None

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path(__file__).resolve().parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._sqlalchemy_handler = self._configure_sqlalchemy_logging()
        self.execution_logger = self._configure_execution_logger()

    @classmethod
    def get_instance(cls) -> "SQLLogger":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_execution_logger(self) -> logging.Logger:
        return self.execution_logger

    def _configure_sqlalchemy_logging(self) -> logging.Handler:
        handler = logging.FileHandler(self.log_dir / "sqlalchemy.log", mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

        for name in ("sqlalchemy.engine", "sqlalchemy.pool"):
            logger = logging.getLogger(name)
            logger.handlers.clear()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False

        return handler

    def _configure_execution_logger(self) -> logging.Logger:
        logger = logging.getLogger("query_execution")
        logger.handlers.clear()

        handler = logging.FileHandler(self.log_dir / "execution.log", mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        return logger