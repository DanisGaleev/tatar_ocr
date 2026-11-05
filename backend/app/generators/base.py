from abc import ABC, abstractmethod
import random
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
    def generate(self, seed: Optional[int] = None, **kwargs: Any) -> TaskDraft:
        """Generate a single deterministic task if seed is given, or pseudo-random if None."""
        pass

    def generate_batch(self, count: int, seed: Optional[int] = None, **kwargs: Any) -> List[TaskDraft]:
        """Generate a batch of non-duplicate tasks deterministically."""
        rng = random.Random(seed) if seed is not None else random.Random()
        drafts: List[TaskDraft] = []
        seen: set[tuple[str, str]] = set()
        for _ in range(count * 6):
            if len(drafts) >= count:
                break
            item_seed = rng.randint(0, 10_000_000)
            draft = self.generate(seed=item_seed, **kwargs)
            key = (draft.prompt_tt, draft.expected_answer)
            if key not in seen:
                seen.add(key)
                drafts.append(draft)
        while len(drafts) < count:
            item_seed = rng.randint(0, 10_000_000)
            drafts.append(self.generate(seed=item_seed, **kwargs))
        return drafts

    def create_custom(self, **kwargs: Any) -> TaskDraft:
        """Create a validated task from custom teacher parameters."""
        raise NotImplementedError(f"Custom creation not supported for {self.task_type}")
