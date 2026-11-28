import random
from typing import Optional, List, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.russian.phonetics import clean_russian_word, mask_orthogram_combination
from app.generators.russian.lexicon import (
    ORTHOGRAM_COMBINATION_WORDS,
    UNSTRESSED_VOWEL_ITEMS,
    PAIRED_CONSONANT_ITEMS,
    UNPRONOUNCEABLE_CONSONANT_ITEMS,
    SOFT_HARD_SIGN_ITEMS,
    DICTIONARY_WORDS_ITEMS,
)


class RuOrthogramCombinationsGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_orthogram_combinations"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "orthography"

    @property
    def default_topic_tag(self) -> str:
        return "ru_orthogram_combinations"

    @property
    def topic_name_tt(self) -> str:
        return "Правописание ЖИ–ШИ, ЧА–ЩА, ЧУ–ЩУ"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 1,
        stems: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = stems or ORTHOGRAM_COMBINATION_WORDS
        word = rng.choice(pool)
        masked, rule = mask_orthogram_combination(word)
        ans = clean_russian_word(word).upper()
        cell_count = max(8, len(ans))
        if cell_count > 12:
            raise ValueError(f"Generated answer '{ans}' exceeds 12-cell A4 limit.")

        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Вставьте букву и запишите слово целиком ({rule}): {masked} ->",
            expected_answer=ans,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"word": word, "masked": masked, "rule": rule},
        )

    def create_custom(self, word: str, grade_level: int = 1, **kwargs: Any) -> TaskDraft:
        masked, rule = mask_orthogram_combination(word)
        ans = clean_russian_word(word).upper()
        cell_count = max(8, len(ans))
        if cell_count > 12:
            raise ValueError(f"Answer '{ans}' exceeds 12-cell A4 limit.")
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Вставьте букву и запишите слово целиком ({rule}): {masked} ->",
            expected_answer=ans,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"word": word, "masked": masked, "rule": rule, "is_custom": True},
        )


class RuUnstressedVowelGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_unstressed_vowel"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "orthography"

    @property
    def default_topic_tag(self) -> str:
        return "ru_unstressed_vowel"

    @property
    def topic_name_tt(self) -> str:
        return "Безударная гласная в корне слова"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 2,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        eligible = [it for it in UNSTRESSED_VOWEL_ITEMS if it[4] <= max(1, grade_level)] or UNSTRESSED_VOWEL_ITEMS
        word_gap, check_word, missing_letter, full_word, _ = rng.choice(eligible)
        ans = clean_russian_word(missing_letter).upper()
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Вставьте безударную гласную (проверочное слово: {check_word}): {word_gap} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word_gap": word_gap, "check_word": check_word, "full_word": full_word},
        )

    def create_custom(
        self,
        word_gap: str,
        check_word: str,
        missing_letter: str,
        grade_level: int = 2,
        **kwargs: Any,
    ) -> TaskDraft:
        ans = clean_russian_word(missing_letter).upper()
        if len(ans) != 1:
            raise ValueError("Missing vowel must be exactly 1 letter.")
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Вставьте безударную гласную (проверочное слово: {check_word.strip()}): {word_gap.strip()} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word_gap": word_gap.strip(), "check_word": check_word.strip(), "is_custom": True},
        )


class RuPairedConsonantGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_paired_consonant"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "orthography"

    @property
    def default_topic_tag(self) -> str:
        return "ru_paired_consonant"

    @property
    def topic_name_tt(self) -> str:
        return "Парная согласная в корне слова"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 2,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        eligible = [it for it in PAIRED_CONSONANT_ITEMS if it[4] <= max(2, grade_level)] or PAIRED_CONSONANT_ITEMS
        word_gap, check_word, missing_letter, full_word, _ = rng.choice(eligible)
        ans = clean_russian_word(missing_letter).upper()
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Вставьте парную согласную (проверочное слово: {check_word}): {word_gap} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word_gap": word_gap, "check_word": check_word, "full_word": full_word},
        )

    def create_custom(
        self,
        word_gap: str,
        check_word: str,
        missing_letter: str,
        grade_level: int = 2,
        **kwargs: Any,
    ) -> TaskDraft:
        ans = clean_russian_word(missing_letter).upper()
        if len(ans) != 1:
            raise ValueError("Missing consonant must be exactly 1 letter.")
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Вставьте парную согласную (проверочное слово: {check_word.strip()}): {word_gap.strip()} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word_gap": word_gap.strip(), "check_word": check_word.strip(), "is_custom": True},
        )


class RuUnpronounceableConsonantGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_unpronounceable_consonant"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "orthography"

    @property
    def default_topic_tag(self) -> str:
        return "ru_unpronounceable_consonant"

    @property
    def topic_name_tt(self) -> str:
        return "Непроизносимая согласная в корне"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 3,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        word_gap, check_word, missing_letter, full_word = rng.choice(UNPRONOUNCEABLE_CONSONANT_ITEMS)
        ans = clean_russian_word(missing_letter).upper()
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Вставьте непроизносимую согласную ({check_word}): {word_gap} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word_gap": word_gap, "check_word": check_word, "full_word": full_word},
        )

    def create_custom(
        self,
        word_gap: str,
        check_word: str,
        missing_letter: str,
        grade_level: int = 3,
        **kwargs: Any,
    ) -> TaskDraft:
        ans = clean_russian_word(missing_letter).upper()
        if len(ans) != 1:
            raise ValueError("Missing consonant must be exactly 1 letter.")
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Вставьте непроизносимую согласную ({check_word.strip()}): {word_gap.strip()} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word_gap": word_gap.strip(), "check_word": check_word.strip(), "is_custom": True},
        )


class RuSoftHardSignGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_soft_hard_sign"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "orthography"

    @property
    def default_topic_tag(self) -> str:
        return "ru_soft_hard_sign"

    @property
    def topic_name_tt(self) -> str:
        return "Разделительные Ъ и Ь знаки"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 3,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        word_gap, missing_sign, full_word = rng.choice(SOFT_HARD_SIGN_ITEMS)
        ans = clean_russian_word(missing_sign).upper()
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Вставьте разделительный Ъ или Ь знак: {word_gap} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word_gap": word_gap, "full_word": full_word},
        )

    def create_custom(self, word_gap: str, missing_sign: str, grade_level: int = 3, **kwargs: Any) -> TaskDraft:
        ans = clean_russian_word(missing_sign).upper()
        if ans not in ("Ъ", "Ь"):
            raise ValueError("Separating sign must be 'Ъ' or 'Ь'.")
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Вставьте разделительный Ъ или Ь знак: {word_gap.strip()} ->",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word_gap": word_gap.strip(), "is_custom": True},
        )


class RuDictionaryWordsGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_dictionary_words"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "orthography"

    @property
    def default_topic_tag(self) -> str:
        return "ru_dictionary_words"

    @property
    def topic_name_tt(self) -> str:
        return "Словарные слова"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 2,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        eligible = [it for it in DICTIONARY_WORDS_ITEMS if it[2] <= max(1, grade_level)] or DICTIONARY_WORDS_ITEMS
        word_gap, full_word, _ = rng.choice(eligible)
        ans = clean_russian_word(full_word).upper()
        cell_count = max(8, len(ans))
        if cell_count > 12:
            raise ValueError(f"Generated answer '{ans}' exceeds 12-cell A4 limit.")

        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Запишите словарное слово полностью: {word_gap} ->",
            expected_answer=ans,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"word_gap": word_gap},
        )

    def create_custom(self, word_gap: str, full_word: str, grade_level: int = 2, **kwargs: Any) -> TaskDraft:
        ans = clean_russian_word(full_word).upper()
        cell_count = max(8, len(ans))
        if cell_count > 12:
            raise ValueError(f"Answer '{ans}' exceeds 12-cell A4 limit.")
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Запишите словарное слово полностью: {word_gap.strip()} ->",
            expected_answer=ans,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"word_gap": word_gap.strip(), "is_custom": True},
        )
