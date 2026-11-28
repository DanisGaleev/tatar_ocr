import random
import re
from typing import Optional, List, Dict, Any, Tuple
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.lexicon import (
    TEXT_COMPREHENSION_SETS,
    TEXT_TITLE_SETS,
)

_CYR_LETTERS = ["А", "Б", "В", "Г"]


def _shuffle_mc_options(opts: List[str], answer_letter: str, rng: random.Random) -> Tuple[List[str], str]:
    clean_bodies = [re.sub(r"^[АБВГA-D]\)\s*", "", o.strip()) for o in opts]
    ans_upper = answer_letter.strip().upper()
    orig_idx = _CYR_LETTERS.index(ans_upper) if ans_upper in _CYR_LETTERS else 0
    paired = [(body, i == orig_idx) for i, body in enumerate(clean_bodies)]
    rng.shuffle(paired)
    relabeled = [f"{_CYR_LETTERS[i]}) {body}" for i, (body, _) in enumerate(paired)]
    new_ans = next(_CYR_LETTERS[i] for i, (_, is_ans) in enumerate(paired) if is_ans)
    return relabeled, new_ans


class TextComprehensionGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "text_comprehension"

    @property
    def default_topic_tag(self) -> str:
        return "reading_comp"

    @property
    def topic_name_tt(self) -> str:
        return "Текст буенча аңлау"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        item = rng.choice(TEXT_COMPREHENSION_SETS)
        text = item["text"]
        q = item["question"]
        opts, ans = _shuffle_mc_options(item["options"], item["answer"], rng)

        opts_str = "   ".join(opts)
        prompt = f"{text}\n{q} {opts_str}"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"text": text, "question": q, "options": opts},
        )

    def create_custom(self, text: str, question: str, options: List[str], correct_letter: str, grade_level: int = 7, **kwargs: Any) -> TaskDraft:
        opts_str = "   ".join(options)
        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"{text.strip()}\n{question.strip()} {opts_str}",
            expected_answer=correct_letter.strip().upper(),
            cell_count=1,
            grade_level=grade_level,
            metadata={"text": text, "question": question, "options": options, "is_custom": True},
        )


class TextTitleMainIdeaGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "text_title_main_idea"

    @property
    def default_topic_tag(self) -> str:
        return "reading_title"

    @property
    def topic_name_tt(self) -> str:
        return "Текстның төп фикере / исеме"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        item = rng.choice(TEXT_TITLE_SETS)
        text = item["text"]
        q = item["question"]
        opts, ans = _shuffle_mc_options(item["options"], item["answer"], rng)

        opts_str = "   ".join(opts)
        prompt = f"{text}\n{q} {opts_str}"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"text": text, "question": q, "options": opts},
        )

    def create_custom(self, text: str, question: str, options: List[str], correct_letter: str, grade_level: int = 7, **kwargs: Any) -> TaskDraft:
        opts_str = "   ".join(options)
        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"{text.strip()}\n{question.strip()} {opts_str}",
            expected_answer=correct_letter.strip().upper(),
            cell_count=1,
            grade_level=grade_level,
            metadata={"text": text, "question": question, "options": options, "is_custom": True},
        )
