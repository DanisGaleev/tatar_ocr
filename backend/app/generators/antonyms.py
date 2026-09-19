import random
from typing import Optional, List, Dict, Any, Tuple
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.phonetics import clean_word
from app.generators.lexicon import ANTONYM_PAIRS as DEFAULT_ANTONYM_PAIRS


class AntonymsGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "antonyms"

    @property
    def default_topic_tag(self) -> str:
        return "antonyms_adjectives"

    @property
    def topic_name_tt(self) -> str:
        return "Сыйфат антонимнары"

    def generate(
        self,
        seed: Optional[int] = None,
        pairs: Optional[List[Tuple[str, str]]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        pair_pool = pairs or DEFAULT_ANTONYM_PAIRS
        word, antonym = rng.choice(pair_pool)
        
        expected_answer = antonym.upper().strip()
        prompt = f"Антонимны (кире мәгънәне) табыгыз: {word} ->"
        cell_count = max(6, len(expected_answer))
        
        if cell_count > 12:
            raise ValueError(f"Expected answer '{expected_answer}' exceeds maximum A4 cell limit (12).")

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={
                "source_word": word,
                "antonym": expected_answer,
            },
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        pairs: Optional[List[Tuple[str, str]]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pair_pool = list(pairs or DEFAULT_ANTONYM_PAIRS)
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
        word: str,
        antonym: str,
        grade_level: int = 6,
        custom_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> TaskDraft:
        clean_w = clean_word(word)
        clean_a = clean_word(antonym).upper()
        
        prompt = custom_prompt or f"Антонимны (кире мәгънәне) табыгыз: {clean_w} ->"
        cell_count = max(6, len(clean_a))
        
        if cell_count > 12:
            raise ValueError(f"Custom answer '{clean_a}' exceeds maximum A4 cell limit (12).")

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=clean_a,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={
                "source_word": clean_w,
                "antonym": clean_a,
                "is_custom": True,
            },
        )
