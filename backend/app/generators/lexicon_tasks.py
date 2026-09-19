import random
from typing import Optional, List, Dict, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.lexicon import (
    SYNONYM_PAIRS,
    COMPOUND_WORDS,
    ODD_ONE_OUT_SETS,
)


class SynonymsGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "synonyms"

    @property
    def default_topic_tag(self) -> str:
        return "synonyms"

    @property
    def topic_name_tt(self) -> str:
        return "Синонимнар"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        pair = rng.choice(SYNONYM_PAIRS)
        word, synonym = pair

        prompt = f"Сүзгә синоним языгыз: {word} ->"
        cell_count = max(8, len(synonym))

        return TaskDraft(
            task_type=self.task_type,
            topic_tag="synonyms",
            topic_name_tt="Синонимнар",
            prompt_tt=prompt,
            expected_answer=synonym,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"base_word": word},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = list(SYNONYM_PAIRS)
        rng.shuffle(pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item_seed = rng.randint(0, 10_000_000)
            pair = pool[i % len(pool)]
            draft = TaskDraft(
                task_type=self.task_type,
                topic_tag="synonyms",
                topic_name_tt="Синонимнар",
                prompt_tt=f"Сүзгә синоним языгыз: {pair[0]} ->",
                expected_answer=pair[1],
                cell_count=max(8, len(pair[1])),
                grade_level=grade_level,
                metadata={"base_word": pair[0]},
            )
            drafts.append(draft)
        return drafts

    def create_custom(self, base_word: str, expected_synonym: str, grade_level: int = 5, **kwargs: Any) -> TaskDraft:
        clean_syn = expected_synonym.strip().upper()
        prompt = f"Сүзгә синоним языгыз: {base_word.strip()} ->"
        cell_count = max(8, len(clean_syn))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="synonyms",
            topic_name_tt="Синонимнар",
            prompt_tt=prompt,
            expected_answer=clean_syn,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"base_word": base_word.strip(), "is_custom": True},
        )


class CompoundWordsGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "compound_words"

    @property
    def default_topic_tag(self) -> str:
        return "compound_words"

    @property
    def topic_name_tt(self) -> str:
        return "Кушма сүзләр"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        item = rng.choice(COMPOUND_WORDS)
        p1, p2, compound = item

        prompt = f"Кушма сүз ясагыз: {p1} + {p2} ->"
        cell_count = max(8, len(compound))

        return TaskDraft(
            task_type=self.task_type,
            topic_tag="compound_words",
            topic_name_tt="Кушма сүзләр",
            prompt_tt=prompt,
            expected_answer=compound,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"part1": p1, "part2": p2},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = list(COMPOUND_WORDS)
        rng.shuffle(pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item = pool[i % len(pool)]
            draft = TaskDraft(
                task_type=self.task_type,
                topic_tag="compound_words",
                topic_name_tt="Кушма сүзләр",
                prompt_tt=f"Кушма сүз ясагыз: {item[0]} + {item[1]} ->",
                expected_answer=item[2],
                cell_count=max(8, len(item[2])),
                grade_level=grade_level,
                metadata={"part1": item[0], "part2": item[1]},
            )
            drafts.append(draft)
        return drafts

    def create_custom(self, part1: str, part2: str, expected_compound: str, grade_level: int = 6, **kwargs: Any) -> TaskDraft:
        clean_comp = expected_compound.strip().upper()
        prompt = f"Кушма сүз ясагыз: {part1.strip()} + {part2.strip()} ->"
        cell_count = max(8, len(clean_comp))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="compound_words",
            topic_name_tt="Кушма сүзләр",
            prompt_tt=prompt,
            expected_answer=clean_comp,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"part1": part1.strip(), "part2": part2.strip(), "is_custom": True},
        )


class OddOneOutGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "odd_one_out"

    @property
    def default_topic_tag(self) -> str:
        return "odd_one_out"

    @property
    def topic_name_tt(self) -> str:
        return "Артыгын тап"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        item = rng.choice(ODD_ONE_OUT_SETS)
        words, odd_word = item

        words_str = ", ".join(words)
        prompt = f"Артык сүзне табыгыз: {words_str} ->"
        cell_count = max(8, len(odd_word))

        return TaskDraft(
            task_type=self.task_type,
            topic_tag="odd_one_out",
            topic_name_tt="Артыгын тап",
            prompt_tt=prompt,
            expected_answer=odd_word,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"words": words},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = list(ODD_ONE_OUT_SETS)
        rng.shuffle(pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item = pool[i % len(pool)]
            words, odd_word = item
            draft = TaskDraft(
                task_type=self.task_type,
                topic_tag="odd_one_out",
                topic_name_tt="Артыгын тап",
                prompt_tt=f"Артык сүзне табыгыз: {', '.join(words)} ->",
                expected_answer=odd_word,
                cell_count=max(8, len(odd_word)),
                grade_level=grade_level,
                metadata={"words": words},
            )
            drafts.append(draft)
        return drafts

    def create_custom(self, words: List[str], expected_odd_word: str, grade_level: int = 5, **kwargs: Any) -> TaskDraft:
        clean_odd = expected_odd_word.strip().upper()
        prompt = f"Артык сүзне табыгыз: {', '.join(words)} ->"
        cell_count = max(8, len(clean_odd))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="odd_one_out",
            topic_name_tt="Артыгын тап",
            prompt_tt=prompt,
            expected_answer=clean_odd,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"words": words, "is_custom": True},
        )
