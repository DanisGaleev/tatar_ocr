from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from app.generators.phonetics import build_expected_cells

class TaskDraft(BaseModel):
    task_type: str
    topic_tag: str
    topic_name_tt: str
    prompt_tt: str
    expected_answer: str
    cell_count: int = Field(..., ge=1, le=12)
    grade_level: int = Field(default=5, ge=5, le=9)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    expected_cells: List[Dict[str, Any]] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.expected_cells:
            self.expected_cells = build_expected_cells(self.expected_answer, self.cell_count)

class BaseTaskGenerator(ABC):
    @property
    @abstractmethod
    def task_type(self) -> str:
        """Unique identifier of the task type."""
        pass

    @property
    @abstractmethod
    def default_topic_tag(self) -> str:
        """Default topic tag for analytics aggregation."""
        pass

    @property
    @abstractmethod
    def topic_name_tt(self) -> str:
        """Human-readable topic name in Tatar."""
        pass

    @abstractmethod
    def generate(self, seed: Optional[int] = None, **kwargs) -> TaskDraft:
        """Generate a single deterministic task if seed is given, or pseudo-random if None."""
        pass

    @abstractmethod
    def generate_batch(self, count: int, seed: Optional[int] = None, **kwargs) -> List[TaskDraft]:
        """Generate a batch of non-duplicate tasks deterministically."""
        pass

    @abstractmethod
    def create_custom(self, **kwargs) -> TaskDraft:
        """Create a validated task from custom teacher parameters."""
        pass
