import random
from typing import Optional, List, Dict, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.phonetics import (
    CASE_NAMES,
    inflect_case,
    clean_word,
    PhoneticsError,
)
from app.generators.lexicon import NOUN_STEMS as DEFAULT_NOUN_STEMS


class CaseInflectionGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "case_inflection"

    @property
    def default_topic_tag(self) -> str:
        return "case_inflection"

    @property
    def topic_name_tt(self) -> str:
        return "Исем килешләре"

    def generate(
        self,
        seed: Optional[int] = None,
        case_code: Optional[str] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = stems or DEFAULT_NOUN_STEMS
        
        available_cases = list(CASE_NAMES.keys())
        chosen_case = case_code if case_code and case_code in CASE_NAMES else rng.choice(available_cases)
        chosen_stem = rng.choice(stem_pool)
        
        expected_answer = inflect_case(chosen_stem, chosen_case)
        case_name = CASE_NAMES[chosen_case]
        prompt = f"Куегыз сүзне {case_name.lower()} килешендә: {chosen_stem} ->"
        
        cell_count = max(8, len(expected_answer))
        if cell_count > 12:
            raise ValueError(f"Generated answer '{expected_answer}' exceeds maximum A4 cell limit (12).")

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=chosen_case,
            topic_name_tt=f"{case_name} килеше",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={
                "stem": chosen_stem,
                "case_code": chosen_case,
                "case_name": case_name,
            },
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        case_code: Optional[str] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = list(stems or DEFAULT_NOUN_STEMS)
        rng.shuffle(stem_pool)
        
        available_cases = [case_code] if (case_code and case_code in CASE_NAMES) else list(CASE_NAMES.keys())
        
        drafts: List[TaskDraft] = []
        for i in range(count):
            item_seed = rng.randint(0, 10_000_000)
            stem = stem_pool[i % len(stem_pool)]
            case = available_cases[i % len(available_cases)]
            draft = self.generate(
                seed=item_seed,
                case_code=case,
                stems=[stem],
                grade_level=grade_level,
            )
            drafts.append(draft)
            
        return drafts

    def create_custom(
        self,
        stem: str,
        case_code: str,
        grade_level: int = 7,
        custom_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> TaskDraft:
        clean_s = clean_word(stem)
        if case_code not in CASE_NAMES:
            raise ValueError(f"Unknown case_code '{case_code}'. Must be one of: {list(CASE_NAMES.keys())}")
        
        expected_answer = inflect_case(clean_s, case_code)
        case_name = CASE_NAMES[case_code]
        prompt = custom_prompt or f"Куегыз сүзне {case_name.lower()} килешендә: {clean_s} ->"
        
        cell_count = max(8, len(expected_answer))
        if cell_count > 12:
            raise ValueError(f"Generated answer '{expected_answer}' exceeds maximum A4 cell limit (12).")

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=case_code,
            topic_name_tt=f"{case_name} килеше",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={
                "stem": clean_s,
                "case_code": case_code,
                "case_name": case_name,
                "is_custom": True,
            },
        )
