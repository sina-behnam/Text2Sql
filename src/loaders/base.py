from typing import Any

class BaseLoader:

    def __init__(self):
        pass

    def format_data(self, *args, **kwargs) -> Any:
        pass

    def load_data(self, *args, **kwargs) -> Any:
        raise NotImplementedError("load_data method must be implemented by subclasses.")