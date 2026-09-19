import random
from typing import Optional, List, Dict, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.lexicon import (
    TEXT_COMPREHENSION_SETS,
    TEXT_TITLE_SETS,
)


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
        opts = item["options"]
        ans = item["answer"]

        opts_str = "   ".join(opts)
        prompt = f"{text}\n{q} {opts_str}"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag="reading_comp",
            topic_name_tt="Текстны аңлау",
            prompt_tt=prompt,
            expected_answer=ans.upper(),
            cell_count=1,
            grade_level=grade_level,
            metadata={"text": text, "question": q, "options": opts},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = list(TEXT_COMPREHENSION_SETS)
        rng.shuffle(pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item = pool[i % len(pool)]
            text = item["text"]
            q = item["question"]
            opts = item["options"]
            ans = item["answer"]
            opts_str = "   ".join(opts)
            draft = TaskDraft(
                task_type=self.task_type,
                topic_tag="reading_comp",
                topic_name_tt="Текстны аңлау",
                prompt_tt=f"{text}\n{q} {opts_str}",
                expected_answer=ans.upper(),
                cell_count=1,
                grade_level=grade_level,
                metadata={"text": text, "question": q, "options": opts},
            )
            drafts.append(draft)
        return drafts

    def create_custom(self, text: str, question: str, options: List[str], correct_letter: str, grade_level: int = 7, **kwargs: Any) -> TaskDraft:
        opts_str = "   ".join(options)
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="reading_comp",
            topic_name_tt="Текстны аңлау",
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
        opts = item["options"]
        ans = item["answer"]

        opts_str = "   ".join(opts)
        prompt = f"{text}\n{q} {opts_str}"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag="reading_title",
            topic_name_tt="Текстның исеме",
            prompt_tt=prompt,
            expected_answer=ans.upper(),
            cell_count=1,
            grade_level=grade_level,
            metadata={"text": text, "question": q, "options": opts},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = list(TEXT_TITLE_SETS)
        rng.shuffle(pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item = pool[i % len(pool)]
            text = item["text"]
            q = item["question"]
            opts = item["options"]
            ans = item["answer"]
            opts_str = "   ".join(opts)
            draft = TaskDraft(
                task_type=self.task_type,
                topic_tag="reading_title",
                topic_name_tt="Текстның исеме",
                prompt_tt=f"{text}\n{q} {opts_str}",
                expected_answer=ans.upper(),
                cell_count=1,
                grade_level=grade_level,
                metadata={"text": text, "question": q, "options": opts},
            )
            drafts.append(draft)
        return drafts

    def create_custom(self, text: str, question: str, options: List[str], correct_letter: str, grade_level: int = 7, **kwargs: Any) -> TaskDraft:
        opts_str = "   ".join(options)
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="reading_title",
            topic_name_tt="Текстның исеме",
            prompt_tt=f"{text.strip()}\n{question.strip()} {opts_str}",
            expected_answer=correct_letter.strip().upper(),
            cell_count=1,
            grade_level=grade_level,
            metadata={"text": text, "question": question, "options": options, "is_custom": True},
        )
