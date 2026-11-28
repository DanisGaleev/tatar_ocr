import random
from typing import Optional, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.russian.phonetics import clean_russian_word
from app.generators.russian.lexicon import (
    NOUN_GENDER_ITEMS,
    NOUN_NUMBER_ITEMS,
    PARTS_OF_SPEECH_ITEMS,
    NOUN_CASE_ITEMS,
    RU_CASE_LABELS,
)


class RuNounGenderGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_noun_gender"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "morphology"

    @property
    def default_topic_tag(self) -> str:
        return "ru_noun_gender"

    @property
    def topic_name_tt(self) -> str:
        return "Род имён существительных"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 3,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        word, gender_code = rng.choice(NOUN_GENDER_ITEMS)
        ans = clean_russian_word(gender_code).upper()
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Определите род существительного (М — мужской, Ж — женский, С — средний): {word} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word": word, "gender": ans},
        )

    def create_custom(self, word: str, gender_code: str, grade_level: int = 3, **kwargs: Any) -> TaskDraft:
        ans = clean_russian_word(gender_code).upper()
        if ans not in ("М", "Ж", "С"):
            raise ValueError("Gender code must be 'М', 'Ж', or 'С'.")
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Определите род существительного (М — мужской, Ж — женский, С — средний): {clean_russian_word(word)} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word": word.strip(), "gender": ans, "is_custom": True},
        )


class RuNounNumberGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_noun_number"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "morphology"

    @property
    def default_topic_tag(self) -> str:
        return "ru_noun_number"

    @property
    def topic_name_tt(self) -> str:
        return "Число имён существительных"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 2,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        singular, plural = rng.choice(NOUN_NUMBER_ITEMS)
        ans = clean_russian_word(plural).upper()
        cell_count = max(8, len(ans))
        if cell_count > 12:
            raise ValueError(f"Generated answer '{ans}' exceeds 12-cell A4 limit.")

        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Поставьте существительное во множественное число: {singular} ->",
            expected_answer=ans,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"singular": singular},
        )

    def create_custom(self, singular: str, plural: str, grade_level: int = 2, **kwargs: Any) -> TaskDraft:
        ans = clean_russian_word(plural).upper()
        cell_count = max(8, len(ans))
        if cell_count > 12:
            raise ValueError(f"Answer '{ans}' exceeds 12-cell A4 limit.")
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Поставьте существительное во множественное число: {clean_russian_word(singular)} ->",
            expected_answer=ans,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"singular": singular.strip(), "is_custom": True},
        )


class RuPartsOfSpeechGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_parts_of_speech"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "morphology"

    @property
    def default_topic_tag(self) -> str:
        return "ru_parts_of_speech"

    @property
    def topic_name_tt(self) -> str:
        return "Части речи"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 2,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        word, pos_code = rng.choice(PARTS_OF_SPEECH_ITEMS)
        ans = clean_russian_word(pos_code).upper()
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Определите часть речи (С — сущ., П — прил., Г — глагол): {word} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word": word, "pos": ans},
        )

    def create_custom(self, word: str, pos_code: str, grade_level: int = 2, **kwargs: Any) -> TaskDraft:
        ans = clean_russian_word(pos_code).upper()
        if ans not in ("С", "П", "Г"):
            raise ValueError("Part of speech code must be 'С', 'П', or 'Г'.")
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Определите часть речи (С — сущ., П — прил., Г — глагол): {clean_russian_word(word)} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word": word.strip(), "pos": ans, "is_custom": True},
        )


class RuNounCaseGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_noun_case"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "morphology"

    @property
    def default_topic_tag(self) -> str:
        return "ru_noun_case"

    @property
    def topic_name_tt(self) -> str:
        return "Падежи имён существительных"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 4,
        case_code: Optional[str] = None,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = NOUN_CASE_ITEMS
        if case_code:
            code_upper = case_code.strip().upper()
            filtered = [it for it in NOUN_CASE_ITEMS if it[1] == code_upper]
            if filtered:
                pool = filtered
        phrase, c_code = rng.choice(pool)
        ans = clean_russian_word(c_code).upper()
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=f"ru_case_{ans.lower()}",
            topic_name_tt=f"{RU_CASE_LABELS[ans]} падеж",
            prompt_tt=f"Определите падеж выделенного слова (И, Р, Д, В, Т, П):\n«{phrase}»",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"phrase": phrase, "case_code": ans},
        )

    def create_custom(self, phrase: str, case_code: str, grade_level: int = 4, **kwargs: Any) -> TaskDraft:
        ans = clean_russian_word(case_code).upper()
        if ans not in RU_CASE_LABELS:
            raise ValueError(f"Case code must be one of {list(RU_CASE_LABELS.keys())}.")
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=f"ru_case_{ans.lower()}",
            topic_name_tt=f"{RU_CASE_LABELS[ans]} падеж",
            prompt_tt=f"Определите падеж выделенного слова (И, Р, Д, В, Т, П):\n«{phrase.strip()}»",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"phrase": phrase.strip(), "case_code": ans, "is_custom": True},
        )
