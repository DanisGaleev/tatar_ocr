import random
from typing import Optional, List, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.russian.phonetics import clean_russian_word, analyze_russian_phonetics
from app.generators.russian.lexicon import PHONETICS_WORD_POOL, ODD_ONE_OUT_GROUPS


class RuPhoneticsGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_phonetics"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "phonetics"

    @property
    def default_topic_tag(self) -> str:
        return "ru_phonetics"

    @property
    def topic_name_tt(self) -> str:
        return "Фонетика: звуки и буквы, твёрдые и мягкие согласные"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 1,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        analyzed = [analyze_russian_phonetics(w) for w in PHONETICS_WORD_POOL]

        modes = [
            (
                "more_sounds",
                "В каком слове звуков БОЛЬШЕ, чем букв?",
                lambda a: a["sound_count"] > a["letter_count"],
                lambda a: a["sound_count"] <= a["letter_count"],
            ),
            (
                "fewer_sounds",
                "В каком слове звуков МЕНЬШЕ, чем букв?",
                lambda a: a["sound_count"] < a["letter_count"],
                lambda a: a["sound_count"] >= a["letter_count"],
            ),
            (
                "all_hard",
                "В каком слове ВСЕ согласные звуки твёрдые?",
                lambda a: a["all_consonants_hard"],
                lambda a: not a["all_consonants_hard"],
            ),
            (
                "all_soft",
                "В каком слове ВСЕ согласные звуки мягкие?",
                lambda a: a["all_consonants_soft"],
                lambda a: not a["all_consonants_soft"],
            ),
        ]

        mode_id, question, target_pred, dist_pred = rng.choice(modes)
        targets = [a["word"] for a in analyzed if target_pred(a)]
        distractors = [a["word"] for a in analyzed if dist_pred(a)]

        target_word = rng.choice(targets)
        dist_words = rng.sample(distractors, 2)
        options = [target_word] + dist_words
        rng.shuffle(options)

        letters = ["А", "Б", "В"]
        ans_letter = letters[options.index(target_word)]
        opts_str = f"А) {options[0]}   Б) {options[1]}   В) {options[2]}"

        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"{question}\n{opts_str}",
            expected_answer=ans_letter,
            cell_count=1,
            grade_level=grade_level,
            metadata={"mode": mode_id, "target": target_word, "options": options},
        )

    def create_custom(
        self,
        question: str,
        options: List[str],
        correct_letter: str,
        grade_level: int = 1,
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
            prompt_tt=f"{question.strip()}\n{opts_str}",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"options": options, "is_custom": True},
        )


class RuOddOneOutGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "ru_odd_one_out"

    @property
    def language(self) -> str:
        return "ru"

    @property
    def category(self) -> str:
        return "phonetics"

    @property
    def default_topic_tag(self) -> str:
        return "ru_odd_one_out"

    @property
    def topic_name_tt(self) -> str:
        return "Найди лишнее слово (грамматическая классификация)"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 3,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        families = [
            ["мужской род", "женский род", "средний род"],
            ["существительные", "прилагательные", "глаголы"],
        ]
        family = rng.choice(families)
        main_cat, odd_cat = rng.sample(family, 2)

        main_words = rng.sample(ODD_ONE_OUT_GROUPS[main_cat], 2)
        odd_word = rng.choice(ODD_ONE_OUT_GROUPS[odd_cat])

        options = main_words + [odd_word]
        rng.shuffle(options)
        letters = ["А", "Б", "В"]
        ans_letter = letters[options.index(odd_word)]
        opts_str = f"А) {options[0]}   Б) {options[1]}   В) {options[2]}"

        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Найдите лишнее слово (укажите букву А, Б или В):\n{opts_str}",
            expected_answer=ans_letter,
            cell_count=1,
            grade_level=grade_level,
            metadata={"main_category": main_cat, "odd_category": odd_cat, "odd_word": odd_word, "options": options},
        )

    def create_custom(self, options: List[str], correct_letter: str, grade_level: int = 3, **kwargs: Any) -> TaskDraft:
        ans = clean_russian_word(correct_letter).upper()
        if ans not in ("А", "Б", "В"):
            raise ValueError("Correct letter must be 'А', 'Б', or 'В'.")
        opts_str = "   ".join(f"{chr(1040 + i)}) {opt}" for i, opt in enumerate(options[:3]))
        return TaskDraft(
            language="ru",
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Найдите лишнее слово (укажите букву А, Б или В):\n{opts_str}",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"options": options, "is_custom": True},
        )
