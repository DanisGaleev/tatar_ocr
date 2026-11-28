import random
from typing import Optional, List, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.russian.phonetics import clean_russian_word
from app.generators.russian.lexicon import RU_ANTONYM_PAIRS, RU_SYNONYM_SETS


class RuAntonymsGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_antonyms"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "lexicon"

    @property
    def default_topic_tag(self) -> str:
        return "ru_antonyms"

    @property
    def topic_name_tt(self) -> str:
        return "Антонимы (слова с противоположным значением)"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 2,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        word, antonym = rng.choice(RU_ANTONYM_PAIRS)
        ans = clean_russian_word(antonym).upper()
        cell_count = max(8, len(ans))
        if cell_count > 12:
            raise ValueError(f"Generated answer '{ans}' exceeds 12-cell A4 limit.")

        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Подберите антоним (противоположное по смыслу слово): {word} ->",
            expected_answer=ans,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"word": word, "antonym": ans},
        )

    def create_custom(self, word: str, antonym: str, grade_level: int = 2, **kwargs: Any) -> TaskDraft:
        ans = clean_russian_word(antonym).upper()
        cell_count = max(8, len(ans))
        if cell_count > 12:
            raise ValueError(f"Answer '{ans}' exceeds 12-cell A4 limit.")
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Подберите антоним (противоположное по смыслу слово): {clean_russian_word(word)} ->",
            expected_answer=ans,
            cell_count=cell_count,
            grade_level=grade_level,
            metadata={"word": word.strip(), "antonym": ans, "is_custom": True},
        )


class RuSynonymsGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_synonyms"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "lexicon"

    @property
    def default_topic_tag(self) -> str:
        return "ru_synonyms"

    @property
    def topic_name_tt(self) -> str:
        return "Синонимы (близкие по смыслу слова)"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 2,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        word, correct_syn, distractors = rng.choice(RU_SYNONYM_SETS)

        options = [correct_syn] + list(distractors[:2])
        rng.shuffle(options)
        letters = ["А", "Б", "В"]
        ans_letter = letters[options.index(correct_syn)]

        opts_str = f"А) {options[0]}   Б) {options[1]}   В) {options[2]}"
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Выберите синоним к слову «{word}»:\n{opts_str}",
            expected_answer=ans_letter,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word": word, "correct_synonym": correct_syn, "options": options},
        )

    def create_custom(
        self,
        word: str,
        options: List[str],
        correct_letter: str,
        grade_level: int = 2,
        **kwargs: Any,
    ) -> TaskDraft:
        ans = clean_russian_word(correct_letter).upper()
        if ans not in ("А", "Б", "В"):
            raise ValueError("Correct letter must be 'А', 'Б', or 'В'.")
        opts_str = "   ".join(f"{chr(1040 + i)}) {opt}" for i, opt in enumerate(options[:3]))
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Выберите синоним к слову «{clean_russian_word(word)}»:\n{opts_str}",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"word": word.strip(), "options": options, "is_custom": True},
        )
