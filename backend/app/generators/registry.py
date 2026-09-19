import json
import random
from typing import Dict, List, Optional, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.case_inflection import CaseInflectionGenerator
from app.generators.plural_affixes import PluralAffixesGenerator
from app.generators.antonyms import AntonymsGenerator
from app.generators.translation import TranslationGenerator
from app.generators.morphology import (
    PossessiveAffixesGenerator,
    PossessiveCaseGenerator,
    VerbNegationGenerator,
    VerbImperativeGenerator,
    ComparativeDegreeGenerator,
    OrdinalNumeralsGenerator,
)
from app.generators.word_formation import (
    NounProfessionGenerator,
    AbstractNounGenerator,
    ConsonantVoicingGenerator,
)
from app.generators.lexicon_tasks import (
    SynonymsGenerator,
    CompoundWordsGenerator,
    OddOneOutGenerator,
)
from app.generators.phonetics_tasks import (
    VowelHarmonyGapGenerator,
    TatarLettersGapGenerator,
)
from app.generators.syntax_tasks import (
    SentenceMCClozeGenerator,
    TrueFalseGrammarGenerator,
    FindErrorSentenceGenerator,
    WordOrderGenerator,
)
from app.generators.reading_tasks import (
    TextComprehensionGenerator,
    TextTitleMainIdeaGenerator,
)
from app.core.config import settings

class GeneratorRegistry:
    def __init__(self) -> None:
        self._generators: Dict[str, BaseTaskGenerator] = {
            # 1. Basic & Existing
            "case_inflection": CaseInflectionGenerator(),
            "plural_affixes": PluralAffixesGenerator(),
            "antonyms": AntonymsGenerator(),
            "translation": TranslationGenerator(),
            # 2. Morphology
            "possessive_affixes": PossessiveAffixesGenerator(),
            "possessive_case": PossessiveCaseGenerator(),
            "verb_negation": VerbNegationGenerator(),
            "verb_imperative": VerbImperativeGenerator(),
            "comparative_degree": ComparativeDegreeGenerator(),
            "ordinal_numerals": OrdinalNumeralsGenerator(),
            # 3. Word Formation
            "noun_profession": NounProfessionGenerator(),
            "abstract_noun": AbstractNounGenerator(),
            "consonant_voicing": ConsonantVoicingGenerator(),
            # 4. Lexicon
            "synonyms": SynonymsGenerator(),
            "compound_words": CompoundWordsGenerator(),
            "odd_one_out": OddOneOutGenerator(),
            # 5. Phonetics & Orthography
            "vowel_harmony_gap": VowelHarmonyGapGenerator(),
            "tatar_letters_gap": TatarLettersGapGenerator(),
            # 6. Syntax & Sentence
            "sentence_mc_cloze": SentenceMCClozeGenerator(),
            "true_false_grammar": TrueFalseGrammarGenerator(),
            "find_error_sentence": FindErrorSentenceGenerator(),
            "word_order": WordOrderGenerator(),
            # 7. Reading
            "text_comprehension": TextComprehensionGenerator(),
            "text_title_main_idea": TextTitleMainIdeaGenerator(),
        }

    def get_generator(self, task_type: str) -> BaseTaskGenerator:
        if task_type not in self._generators:
            raise ValueError(f"Unknown task type '{task_type}'. Available types: {list(self._generators.keys())}")
        return self._generators[task_type]

    def list_supported_types(self) -> List[Dict[str, str]]:
        return [
            {
                "task_type": gen.task_type,
                "topic_tag": gen.default_topic_tag,
                "topic_name_tt": gen.topic_name_tt,
            }
            for gen in self._generators.values()
        ]

    def generate_single(
        self,
        task_type: str,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> TaskDraft:
        gen = self.get_generator(task_type)
        return gen.generate(seed=seed, **kwargs)

    def assemble_variants(
        self,
        assignment_id: str,
        title: str,
        grade_level: int,
        variants_count: int = 1,
        seed: Optional[int] = None,
        task_types: Optional[List[str]] = None,
        questions_per_variant: int = 8,
        shuffle: bool = True,
    ) -> Dict[str, Any]:
        """
        Assembles a full modular test with offline bundle data for mobile grading.
        Enforces maximum questions limit (A4 physical constraint <= 8).
        """
        if questions_per_variant > settings.MAX_QUESTIONS_PER_PAGE:
            raise ValueError(
                f"Requested {questions_per_variant} questions per variant exceeds A4 physical page limit ({settings.MAX_QUESTIONS_PER_PAGE})."
            )

        types_to_use = task_types or list(self._generators.keys())
        base_seed = seed if seed is not None else random.randint(1, 10_000_000)
        
        variants_data: List[Dict[str, Any]] = []

        for v_idx in range(1, variants_count + 1):
            var_seed = base_seed + (v_idx * 1337)
            rng = random.Random(var_seed)
            
            # Select question types cyclically or randomly
            chosen_types = [types_to_use[i % len(types_to_use)] for i in range(questions_per_variant)]
            if shuffle:
                rng.shuffle(chosen_types)
                
            questions: List[Dict[str, Any]] = []
            for q_idx, q_type in enumerate(chosen_types, start=1):
                marker_id = 10 + q_idx
                q_seed = rng.randint(1, 10_000_000)
                gen = self.get_generator(q_type)
                draft = gen.generate(seed=q_seed, grade_level=grade_level)
                
                questions.append({
                    "question_number": q_idx,
                    "marker_id": marker_id,
                    "prompt": draft.prompt_tt,
                    "topic_tag": draft.topic_tag,
                    "topic_name_tt": draft.topic_name_tt,
                    "cell_count": draft.cell_count,
                    "expected_answer": draft.expected_answer,
                    "expected_cells": draft.expected_cells,
                })
                
            qr_signature = json.dumps({
                "tid": assignment_id,
                "var": v_idx,
                "page": 1,
                "tot": 1,
                "n_q": questions_per_variant,
            }, separators=(',', ':'))

            variants_data.append({
                "variant_id": v_idx,
                "qr_signature": qr_signature,
                "template_geometry": {
                    "format": "A4",
                    "corner_aruco_dict": settings.ARUCO_DICT,
                    "corner_aruco_ids": settings.CORNER_ARUCO_IDS,
                    "cell_dimensions_mm": {
                        "width": settings.CELL_WIDTH_MM,
                        "height": settings.CELL_HEIGHT_MM,
                    },
                },
                "questions": questions,
            })

        return {
            "assignment_id": assignment_id,
            "title": title,
            "total_variants": variants_count,
            "variants": variants_data,
        }

registry = GeneratorRegistry()
