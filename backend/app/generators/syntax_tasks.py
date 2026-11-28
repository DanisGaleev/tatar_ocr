import random
from typing import Optional, List, Dict, Any, Union
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.lexicon import (
    SENTENCE_CLOZE_SETS,
    TRUE_FALSE_SETS,
    FIND_ERROR_SETS,
    WORD_ORDER_SETS,
)

_CYR_LETTERS = ["А", "Б", "В", "Г"]


class SentenceMCClozeGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "sentence_mc_cloze"

    @property
    def default_topic_tag(self) -> str:
        return "sentence_cloze"

    @property
    def topic_name_tt(self) -> str:
        return "Җөмләдә кушымчаны сайлау"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        item = rng.choice(SENTENCE_CLOZE_SETS)
        sent, opts, ans = item

        orig_idx = _CYR_LETTERS.index(ans.upper()) if ans.upper() in _CYR_LETTERS else 0
        paired = [(opt, i == orig_idx) for i, opt in enumerate(opts[:3])]
        rng.shuffle(paired)
        shuffled_opts = [p[0] for p in paired]
        new_ans = next(_CYR_LETTERS[i] for i, (_, is_ans) in enumerate(paired) if is_ans)

        opts_str = "   ".join(f"{_CYR_LETTERS[i]}) {opt}" for i, opt in enumerate(shuffled_opts))
        prompt = f"«{sent}»\nДөрес кушымчаны сайлагыз: {opts_str}"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=new_ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"sentence": sent, "options": shuffled_opts},
        )

    def create_custom(self, sentence: str, options: List[str], correct_letter: str, grade_level: int = 7, **kwargs: Any) -> TaskDraft:
        opts_str = "   ".join(f"{_CYR_LETTERS[i]}) {opt}" for i, opt in enumerate(options[:3]))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"«{sentence.strip()}»\nДөрес кушымчаны сайлагыз: {opts_str}",
            expected_answer=correct_letter.strip().upper(),
            cell_count=1,
            grade_level=grade_level,
            metadata={"sentence": sentence, "options": options, "is_custom": True},
        )


class TrueFalseGrammarGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "true_false_grammar"

    @property
    def default_topic_tag(self) -> str:
        return "true_false"

    @property
    def topic_name_tt(self) -> str:
        return "Дөрес яки ялгыш кагыйдә"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        item = rng.choice(TRUE_FALSE_SETS)
        stmt, is_true = item

        ans = "Д" if is_true else "Я"
        prompt = f"{stmt} — Раслау дөресме? (Д — дөрес, Я — ялгыш)"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"statement": stmt, "is_true": is_true},
        )

    def create_custom(self, statement: str, is_true: bool, grade_level: int = 6, **kwargs: Any) -> TaskDraft:
        ans = "Д" if is_true else "Я"
        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"{statement.strip()} — Раслау дөресме? (Д — дөрес, Я — ялгыш)",
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"statement": statement.strip(), "is_true": is_true, "is_custom": True},
        )


class FindErrorSentenceGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "find_error_sentence"

    @property
    def default_topic_tag(self) -> str:
        return "find_error"

    @property
    def topic_name_tt(self) -> str:
        return "Хатаны тап (җөмлә хәрефен)"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        item = rng.choice(FIND_ERROR_SETS)
        sentences, err_idx = item

        paired = [(s, i == err_idx) for i, s in enumerate(sentences[:4])]
        rng.shuffle(paired)
        labeled = [f"{_CYR_LETTERS[i]}) {s}" for i, (s, _) in enumerate(paired)]
        ans_letter = next(_CYR_LETTERS[i] for i, (_, is_err) in enumerate(paired) if is_err)

        sents_formatted = "   ".join(labeled)
        prompt = f"Кайсы җөмләдә грамматик хата бар? (А–Г языгыз): {sents_formatted}"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=ans_letter,
            cell_count=1,
            grade_level=grade_level,
            metadata={"sentences": labeled},
        )

    def create_custom(
        self,
        sentences: List[str],
        error_number: Union[int, str],
        grade_level: int = 7,
        **kwargs: Any,
    ) -> TaskDraft:
        if isinstance(error_number, int):
            if not 1 <= error_number <= 4:
                raise ValueError("error_number must be between 1 and 4.")
            ans_letter = _CYR_LETTERS[error_number - 1]
        else:
            ans_letter = str(error_number).strip().upper()
            if ans_letter not in _CYR_LETTERS:
                raise ValueError("error_number letter must be one of А, Б, В, Г.")

        labeled = [
            s if s.lstrip().startswith(tuple(f"{ltr})" for ltr in _CYR_LETTERS)) else f"{_CYR_LETTERS[i]}) {s}"
            for i, s in enumerate(sentences[:4])
        ]
        sents_formatted = "   ".join(labeled)
        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Кайсы җөмләдә грамматик хата бар? (А–Г языгыз): {sents_formatted}",
            expected_answer=ans_letter,
            cell_count=1,
            grade_level=grade_level,
            metadata={"sentences": labeled, "is_custom": True},
        )


class WordOrderGenerator(BaseTaskGenerator):
    @property
    def task_type(self) -> str:
        return "word_order"

    @property
    def default_topic_tag(self) -> str:
        return "word_order"

    @property
    def topic_name_tt(self) -> str:
        return "Җөмлә төзе (сүзләр тәртибе)"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        item = rng.choice(WORD_ORDER_SETS)
        parts, ans_order = item

        parts_formatted = "   ".join(parts)
        prompt = f"Җөмлә төзегез (хәрефләр тәртибен языгыз): {parts_formatted}"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=prompt,
            expected_answer=ans_order.upper(),
            cell_count=len(ans_order),
            grade_level=grade_level,
            metadata={"parts": parts},
        )

    def create_custom(self, parts: List[str], correct_order: str, grade_level: int = 6, **kwargs: Any) -> TaskDraft:
        parts_formatted = "   ".join(parts)
        clean_order = correct_order.strip().upper()
        return TaskDraft(
            task_type=self.task_type,
            topic_tag=self.default_topic_tag,
            topic_name_tt=self.topic_name_tt,
            prompt_tt=f"Җөмлә төзегез (хәрефләр тәртибен языгыз): {parts_formatted}",
            expected_answer=clean_order,
            cell_count=len(clean_order),
            grade_level=grade_level,
            metadata={"parts": parts, "is_custom": True},
        )
