import random
from typing import Optional, List, Dict, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.phonetics import (
    inflect_plural,
    clean_word,
    get_plural_suffix,
)
from app.generators.lexicon import PLURAL_STEMS as DEFAULT_PLURAL_STEMS



class PluralAffixesGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "plural_affixes"

    @property
    def default_topic_tag(self) -> str:
        return "plural_affixes"

    @property
    def topic_name_tt(self) -> str:
        return "Күплек сан"

    def generate(
        self,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = stems or DEFAULT_PLURAL_STEMS
        chosen_stem = rng.choice(stem_pool)
        
        expected_answer = inflect_plural(chosen_stem)
        suffix = get_plural_suffix(chosen_stem)
        prompt = f"Сүзгә күплек сан кушымчасын ялгагыз: {chosen_stem} ->"
        
        cell_count = max(8, len(expected_answer))
        if cell_count > 12:
            raise ValueError(f"Generated answer '{expected_answer}' exceeds maximum A4 cell limit (12).")

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={
                "stem": chosen_stem,
                "suffix": suffix,
                "is_nasal_rule": suffix in ("нар", "нәр"),
            },
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 5,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = list(stems or DEFAULT_PLURAL_STEMS)
        rng.shuffle(stem_pool)
        
        drafts: List[TaskDraft] = []
        for i in range(count):
            item_seed = rng.randint(0, 10_000_000)
            stem = stem_pool[i % len(stem_pool)]
            draft = self.generate(
                seed=item_seed,
                stems=[stem],
                grade_level=grade_level,
            )
            drafts.append(draft)
            
        return drafts

    def create_custom(
        self,
        stem: str,
        grade_level: int = 5,
        custom_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> TaskDraft:
        clean_s = clean_word(stem)
        expected_answer = inflect_plural(clean_s)
        suffix = get_plural_suffix(clean_s)
        prompt = custom_prompt or f"Сүзгә күплек сан кушымчасын ялгагыз: {clean_s} ->"
        
        cell_count = max(8, len(expected_answer))
        if cell_count > 12:
            raise ValueError(f"Generated answer '{expected_answer}' exceeds maximum A4 cell limit (12).")

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={
                "stem": clean_s,
                "suffix": suffix,
                "is_nasal_rule": suffix in ("нар", "нәр"),
                "is_custom": True,
            },
        )
