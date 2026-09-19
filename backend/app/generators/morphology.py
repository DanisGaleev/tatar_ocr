import random
from typing import Optional, List, Dict, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.phonetics import (
    CASE_NAMES,
    clean_word,
    inflect_possessive,
    inflect_possessive_case,
    inflect_verb_negation,
    inflect_verb_imperative,
    inflect_comparative,
    inflect_ordinal_numeral,
)
from app.generators.lexicon import (
    NOUN_STEMS,
    VERB_STEMS,
    ADJECTIVE_STEMS,
    NUMERAL_STEMS,
)

PERSON_LABELS = {
    1: "I зат (минем)",
    2: "II зат (синең)",
    3: "III зат (аның)",
}


class PossessiveAffixesGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "possessive_affixes"

    @property
    def default_topic_tag(self) -> str:
        return "possessive"

    @property
    def topic_name_tt(self) -> str:
        return "Тартым категориясе"

    def generate(
        self,
        seed: Optional[int] = None,
        person: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = stems or NOUN_STEMS
        chosen_person = person if person in (1, 2, 3) else rng.choice([1, 2, 3])
        chosen_stem = rng.choice(stem_pool)

        expected_answer = inflect_possessive(chosen_stem, person=chosen_person)
        person_lbl = PERSON_LABELS[chosen_person]
        prompt = f"Тартым кушымчасын ялгагыз ({person_lbl}): {chosen_stem} ->"

        cell_count = max(8, len(expected_answer))
        if cell_count > 12:
            raise ValueError(f"Generated answer '{expected_answer}' exceeds maximum A4 cell limit (12).")

        return TaskDraft(
            task_type=self.task_type,
            topic_tag="possessive",
            topic_name_tt="Тартым категориясе",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={
                "stem": chosen_stem,
                "person": chosen_person,
            },
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        person: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = list(stems or NOUN_STEMS)
        rng.shuffle(stem_pool)
        persons = [person] if person in (1, 2, 3) else [1, 2, 3]

        drafts: List[TaskDraft] = []
        for i in range(count):
            item_seed = rng.randint(0, 10_000_000)
            stem = stem_pool[i % len(stem_pool)]
            p = persons[i % len(persons)]
            draft = self.generate(seed=item_seed, person=p, stems=[stem], grade_level=grade_level)
            drafts.append(draft)
        return drafts

    def create_custom(self, stem: str, person: int = 1, grade_level: int = 6, **kwargs: Any) -> TaskDraft:
        clean_s = clean_word(stem)
        expected_answer = inflect_possessive(clean_s, person=person)
        person_lbl = PERSON_LABELS.get(person, f"{person} зат")
        prompt = f"Тартым кушымчасын ялгагыз ({person_lbl}): {clean_s} ->"
        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="possessive",
            topic_name_tt="Тартым категориясе",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": clean_s, "person": person, "is_custom": True},
        )


class PossessiveCaseGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "possessive_case"

    @property
    def default_topic_tag(self) -> str:
        return "possessive_case"

    @property
    def topic_name_tt(self) -> str:
        return "Тартым һәм килеш ялгануы"

    def generate(
        self,
        seed: Optional[int] = None,
        person: Optional[int] = None,
        case_code: Optional[str] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = stems or NOUN_STEMS
        chosen_person = person if person in (1, 2, 3) else rng.choice([1, 2, 3])
        available_cases = list(CASE_NAMES.keys())
        chosen_case = case_code if case_code in CASE_NAMES else rng.choice(available_cases)
        chosen_stem = rng.choice(stem_pool)

        expected_answer = inflect_possessive_case(chosen_stem, person=chosen_person, case_code=chosen_case)
        person_lbl = PERSON_LABELS[chosen_person]
        case_name = CASE_NAMES[chosen_case]
        prompt = f"Тиешле кушымчаларны ялгагыз: {chosen_stem} + {person_lbl} + {case_name.lower()} ->"

        cell_count = max(8, len(expected_answer))
        if cell_count > 12:
            raise ValueError(f"Generated answer '{expected_answer}' exceeds maximum A4 cell limit (12).")

        return TaskDraft(
            task_type=self.task_type,
            topic_tag="possessive_case",
            topic_name_tt="Тартым һәм килеш",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={
                "stem": chosen_stem,
                "person": chosen_person,
                "case_code": chosen_case,
            },
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        person: Optional[int] = None,
        case_code: Optional[str] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = list(stems or NOUN_STEMS)
        rng.shuffle(stem_pool)
        persons = [person] if person in (1, 2, 3) else [1, 2, 3]
        cases = [case_code] if case_code in CASE_NAMES else list(CASE_NAMES.keys())

        drafts: List[TaskDraft] = []
        for i in range(count):
            item_seed = rng.randint(0, 10_000_000)
            stem = stem_pool[i % len(stem_pool)]
            p = persons[i % len(persons)]
            c = cases[i % len(cases)]
            draft = self.generate(seed=item_seed, person=p, case_code=c, stems=[stem], grade_level=grade_level)
            drafts.append(draft)
        return drafts

    def create_custom(
        self, stem: str, person: int = 1, case_code: str = "case_locative", grade_level: int = 7, **kwargs: Any
    ) -> TaskDraft:
        clean_s = clean_word(stem)
        expected_answer = inflect_possessive_case(clean_s, person=person, case_code=case_code)
        person_lbl = PERSON_LABELS.get(person, f"{person} зат")
        case_name = CASE_NAMES.get(case_code, "килеш")
        prompt = f"Тиешле кушымчаларны ялгагыз: {clean_s} + {person_lbl} + {case_name.lower()} ->"
        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="possessive_case",
            topic_name_tt="Тартым һәм килеш",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": clean_s, "person": person, "case_code": case_code, "is_custom": True},
        )


class VerbNegationGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "verb_negation"

    @property
    def default_topic_tag(self) -> str:
        return "verb_negation"

    @property
    def topic_name_tt(self) -> str:
        return "Юклык фигыль"

    def generate(
        self,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = stems or VERB_STEMS
        chosen_stem = rng.choice(stem_pool)

        expected_answer = inflect_verb_negation(chosen_stem)
        prompt = f"Фигыльнең юклык формасын ясагыз (-мый/-ми): {chosen_stem} ->"

        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="verb_negation",
            topic_name_tt="Юклык фигыль",
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
        stem_pool = list(stems or VERB_STEMS)
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
        expected_answer = inflect_verb_negation(clean_s)
        prompt = f"Фигыльнең юклык формасын ясагыз (-мый/-ми): {clean_s} ->"
        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="verb_negation",
            topic_name_tt="Юклык фигыль",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": clean_s, "is_custom": True},
        )


class VerbImperativeGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "verb_imperative"

    @property
    def default_topic_tag(self) -> str:
        return "verb_imperative"

    @property
    def topic_name_tt(self) -> str:
        return "Боерык фигыль"

    def generate(
        self,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = stems or VERB_STEMS
        chosen_stem = rng.choice(stem_pool)

        expected_answer = inflect_verb_imperative(chosen_stem)
        prompt = f"Боерык фигыль куегыз (II зат, күплек: -гыз/-гез): {chosen_stem} ->"

        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="verb_imperative",
            topic_name_tt="Боерык фигыль",
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
        stem_pool = list(stems or VERB_STEMS)
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
        expected_answer = inflect_verb_imperative(clean_s)
        prompt = f"Боерык фигыль куегыз (II зат, күплек: -гыз/-гез): {clean_s} ->"
        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="verb_imperative",
            topic_name_tt="Боерык фигыль",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": clean_s, "is_custom": True},
        )


class ComparativeDegreeGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "comparative_degree"

    @property
    def default_topic_tag(self) -> str:
        return "comparative"

    @property
    def topic_name_tt(self) -> str:
        return "Чагыштыру дәрәҗәсе"

    def generate(
        self,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = stems or ADJECTIVE_STEMS
        chosen_stem = rng.choice(stem_pool)

        expected_answer = inflect_comparative(chosen_stem)
        prompt = f"Сыйфатның чагыштыру дәрәҗәсен ясагыз (-рак/-рәк): {chosen_stem} ->"

        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="comparative",
            topic_name_tt="Чагыштыру дәрәҗәсе",
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
        stem_pool = list(stems or ADJECTIVE_STEMS)
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
        expected_answer = inflect_comparative(clean_s)
        prompt = f"Сыйфатның чагыштыру дәрәҗәсен ясагыз (-рак/-рәк): {clean_s} ->"
        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="comparative",
            topic_name_tt="Чагыштыру дәрәҗәсе",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": clean_s, "is_custom": True},
        )


class OrdinalNumeralsGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ordinal_numerals"

    @property
    def default_topic_tag(self) -> str:
        return "ordinal_numerals"

    @property
    def topic_name_tt(self) -> str:
        return "Тәртип саннары"

    def generate(
        self,
        seed: Optional[int] = None,
        stems: Optional[List[str]] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        stem_pool = stems or NUMERAL_STEMS
        chosen_stem = rng.choice(stem_pool)

        expected_answer = inflect_ordinal_numeral(chosen_stem)
        prompt = f"Тәртип саны кушымчасын ялгагыз (-ынчы/-енче): {chosen_stem} ->"

        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="ordinal_numerals",
            topic_name_tt="Тәртип саннары",
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
        stem_pool = list(stems or NUMERAL_STEMS)
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
        expected_answer = inflect_ordinal_numeral(clean_s)
        prompt = f"Тәртип саны кушымчасын ялгагыз (-ынчы/-енче): {clean_s} ->"
        cell_count = max(8, len(expected_answer))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="ordinal_numerals",
            topic_name_tt="Тәртип саннары",
            prompt_tt=prompt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"stem": clean_s, "is_custom": True},
        )
