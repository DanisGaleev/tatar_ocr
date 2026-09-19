import random
from typing import Optional, List, Dict, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.lexicon import (
    VOWEL_HARMONY_GAPS,
    TATAR_LETTERS_GAPS,
)


class VowelHarmonyGapGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "vowel_harmony_gap"

    @property
    def default_topic_tag(self) -> str:
        return "vowel_harmony"

    @property
    def topic_name_tt(self) -> str:
        return "Сузык авазлар гармониясе"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        item = rng.choice(VOWEL_HARMONY_GAPS)
        gap_prompt, missing_vowel = item

        prompt = f"Төшеп калган сузыкны куегыз: {gap_prompt} ->"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag="vowel_harmony",
            topic_name_tt="Сузык авазлар гармониясе",
            prompt_tt=prompt,
            expected_answer=missing_vowel.upper(),
            cell_count=1,
            grade_level=grade_level,
            metadata={"gap_prompt": gap_prompt},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = list(VOWEL_HARMONY_GAPS)
        rng.shuffle(pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item = pool[i % len(pool)]
            draft = TaskDraft(
                task_type=self.task_type,
                topic_tag="vowel_harmony",
                topic_name_tt="Сузык авазлар гармониясе",
                prompt_tt=f"Төшеп калган сузыкны куегыз: {item[0]} ->",
                expected_answer=item[1].upper(),
                cell_count=1,
                grade_level=grade_level,
                metadata={"gap_prompt": item[0]},
            )
            drafts.append(draft)
        return drafts

    def create_custom(self, gap_prompt: str, missing_vowel: str, grade_level: int = 5, **kwargs: Any) -> TaskDraft:
        clean_v = missing_vowel.strip().upper()
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="vowel_harmony",
            topic_name_tt="Сузык авазлар гармониясе",
            prompt_tt=f"Төшеп калган сузыкны куегыз: {gap_prompt.strip()} ->",
            expected_answer=clean_v,
            cell_count=1,
            grade_level=grade_level,
            metadata={"gap_prompt": gap_prompt.strip(), "is_custom": True},
        )


class TatarLettersGapGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "tatar_letters_gap"

    @property
    def default_topic_tag(self) -> str:
        return "tatar_letters"

    @property
    def topic_name_tt(self) -> str:
        return "Үзенчәлекле татар хәрефләре"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        item = rng.choice(TATAR_LETTERS_GAPS)
        word_gap, missing_char = item

        prompt = f"Татар хәрефен куегыз (Ә, Ө, Ү, Җ, Ң, Һ): {word_gap} ->"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag="tatar_letters",
            topic_name_tt="Татар хәрефләре",
            prompt_tt=prompt,
            expected_answer=missing_char.upper(),
            cell_count=1,
            grade_level=grade_level,
            metadata={"word_gap": word_gap},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = list(TATAR_LETTERS_GAPS)
        rng.shuffle(pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item = pool[i % len(pool)]
            draft = TaskDraft(
                task_type=self.task_type,
                topic_tag="tatar_letters",
                topic_name_tt="Татар хәрефләре",
                prompt_tt=f"Татар хәрефен куегыз (Ә, Ө, Ү, Җ, Ң, Һ): {item[0]} ->",
                expected_answer=item[1].upper(),
                cell_count=1,
                grade_level=grade_level,
                metadata={"word_gap": item[0]},
            )
            drafts.append(draft)
        return drafts

    def create_custom(self, word_gap: str, missing_char: str, grade_level: int = 5, **kwargs: Any) -> TaskDraft:
        clean_ch = missing_char.strip().upper()
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="tatar_letters",
            topic_name_tt="Татар хәрефләре",
            prompt_tt=f"Татар хәрефен куегыз (Ә, Ө, Ү, Җ, Ң, Һ): {word_gap.strip()} ->",
            expected_answer=clean_ch,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word_gap": word_gap.strip(), "is_custom": True},
        )
