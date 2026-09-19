import random
from typing import Optional, List, Dict, Any, Tuple
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.phonetics import clean_word
from app.generators.lexicon import TRANSLATION_PAIRS as DEFAULT_TRANSLATION_PAIRS


class TranslationGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "translation"

    @property
    def default_topic_tag(self) -> str:
        return "vocabulary_translation"

    @property
    def topic_name_tt(self) -> str:
        return "Сүзлек байлыгы (тәрҗемә)"

    def generate(
        self,
        seed: Optional[int] = None,
        pairs: Optional[List[Tuple[str, str]]] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        pair_pool = pairs or DEFAULT_TRANSLATION_PAIRS
        ru_word, tt_word = rng.choice(pair_pool)
        
        expected_answer = tt_word.upper().strip()
        prompt = f"Тәрҗемә итегез ({ru_word}): ->"
        cell_count = max(6, len(expected_answer))
        
        if cell_count > 12:
            raise ValueError(f"Expected translation '{expected_answer}' exceeds maximum A4 cell limit (12).")

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={
                "ru_word": ru_word,
                "tt_word": expected_answer,
            },
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        pairs: Optional[List[Tuple[str, str]]] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pair_pool = list(pairs or DEFAULT_TRANSLATION_PAIRS)
        rng.shuffle(pair_pool)
        
        drafts: List[TaskDraft] = []
        for i in range(count):
            item_seed = rng.randint(0, 10_000_000)
            pair = [pair_pool[i % len(pair_pool)]]
            draft = self.generate(
                seed=item_seed,
                pairs=pair,
                grade_level=grade_level,
            )
            drafts.append(draft)
            
        return drafts

    def create_custom(
        self,
        ru_word: str,
        tt_word: str,
        grade_level: int = 5,
        custom_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> TaskDraft:
        clean_ru = ru_word.strip()
        clean_tt = clean_word(tt_word).upper()
        
        prompt = custom_prompt or f"Тәрҗемә итегез ({clean_ru}): ->"
        cell_count = max(6, len(clean_tt))
        
        if cell_count > 12:
            raise ValueError(f"Custom translation '{clean_tt}' exceeds maximum A4 cell limit (12).")

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=clean_tt,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={
                "ru_word": clean_ru,
                "tt_word": clean_tt,
                "is_custom": True,
            },
        )
