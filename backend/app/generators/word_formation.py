import random
from typing import Optional, List, Dict, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.phonetics import (
    clean_word,
    inflect_profession,
    inflect_abstract_noun,
    inflect_consonant_voicing,
)
from app.generators.lexicon import (
    PROFESSION_STEMS,
    ABSTRACT_NOUN_STEMS,
    VOICING_STEMS,
)


class NounProfessionGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "noun_profession"

    @property
    def default_topic_tag(self) -> str:
        return "noun_profession"

    @property
    def topic_name_tt(self) -> str:
        return "Һөнәр сүзләре (-чы/-че)"

    def generate(
        self,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = stems or PROFESSION_STEMS
        chosen_stem = rng.choice(stem_pool)

        expected_answer = inflect_profession(chosen_stem)
        prompt = f"Һөнәр сүзен ясагыз (-чы/-че): {chosen_stem} ->"

        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="noun_profession",
            topic_name_tt="Һөнәр сүзләре",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": chosen_stem},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = list(stems or PROFESSION_STEMS)
        rng.shuffle(stem_pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item_seed = rng.randint(0, 10_000_000)
            stem = stem_pool[i % len(stem_pool)]
            draft = self.generate(seed=item_seed, stems=[stem], grade_level=grade_level)
            drafts.append(draft)
        return drafts

    def create_custom(self, stem: str, grade_level: int = 6, **kwargs: Any) -> TaskDraft:
        clean_s = clean_word(stem)
        expected_answer = inflect_profession(clean_s)
        prompt = f"Һөнәр сүзен ясагыз (-чы/-че): {clean_s} ->"
        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="noun_profession",
            topic_name_tt="Һөнәр сүзләре",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": clean_s, "is_custom": True},
        )


class AbstractNounGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "abstract_noun"

    @property
    def default_topic_tag(self) -> str:
        return "abstract_noun"

    @property
    def topic_name_tt(self) -> str:
        return "Абстракт исемнәр (-лык/-лек)"

    def generate(
        self,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = stems or ABSTRACT_NOUN_STEMS
        chosen_stem = rng.choice(stem_pool)

        expected_answer = inflect_abstract_noun(chosen_stem)
        prompt = f"Абстракт исем ясагыз (-лык/-лек): {chosen_stem} ->"

        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="abstract_noun",
            topic_name_tt="Абстракт исемнәр",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": chosen_stem},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = list(stems or ABSTRACT_NOUN_STEMS)
        rng.shuffle(stem_pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item_seed = rng.randint(0, 10_000_000)
            stem = stem_pool[i % len(stem_pool)]
            draft = self.generate(seed=item_seed, stems=[stem], grade_level=grade_level)
            drafts.append(draft)
        return drafts

    def create_custom(self, stem: str, grade_level: int = 6, **kwargs: Any) -> TaskDraft:
        clean_s = clean_word(stem)
        expected_answer = inflect_abstract_noun(clean_s)
        prompt = f"Абстракт исем ясагыз (-лык/-лек): {clean_s} ->"
        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="abstract_noun",
            topic_name_tt="Абстракт исемнәр",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": clean_s, "is_custom": True},
        )


class ConsonantVoicingGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "consonant_voicing"

    @property
    def default_topic_tag(self) -> str:
        return "consonant_voicing"

    @property
    def topic_name_tt(self) -> str:
        return "Тартык авазлар алмашынуы"

    def generate(
        self,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = stems or VOICING_STEMS
        chosen_stem = rng.choice(stem_pool)

        expected_answer = inflect_consonant_voicing(chosen_stem)
        prompt = f"Тартык алмашынуын исәпкә алып, III зат тартым куегыз: {chosen_stem} + ы/е ->"

        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="consonant_voicing",
            topic_name_tt="Тартык авазлар алмашынуы",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": chosen_stem},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = list(stems or VOICING_STEMS)
        rng.shuffle(stem_pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item_seed = rng.randint(0, 10_000_000)
            stem = stem_pool[i % len(stem_pool)]
            draft = self.generate(seed=item_seed, stems=[stem], grade_level=grade_level)
            drafts.append(draft)
        return drafts

    def create_custom(self, stem: str, grade_level: int = 7, **kwargs: Any) -> TaskDraft:
        clean_s = clean_word(stem)
        expected_answer = inflect_consonant_voicing(clean_s)
        prompt = f"Тартык алмашынуын исәпкә алып, III зат тартым куегыз: {clean_s} + ы/е ->"
        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="consonant_voicing",
            topic_name_tt="Тартык авазлар алмашынуы",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": clean_s, "is_custom": True},
        )
