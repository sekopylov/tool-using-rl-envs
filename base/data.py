from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Data:
    question: str
    answer: str = ""
    difficulty: int = 1
    initial_state: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None
    gpt_response: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "initial_state": self.initial_state,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Data":
        return cls(
            question=str(payload.get("question", "")),
            answer=str(payload.get("answer", "")),
            difficulty=int(payload.get("difficulty", 1)),
            initial_state=payload.get("initial_state"),
            metadata=payload.get("metadata"),
            gpt_response=str(payload.get("gpt_response", "")),
        )
