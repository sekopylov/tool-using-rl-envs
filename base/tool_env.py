from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from base.data import Data


class ToolEnv(ABC):
    """Multi-step tool-using environment. Text in / text out."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def reset(self, data: Data) -> str:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs: Any,
    ) -> list[Data]:
        raise NotImplementedError
