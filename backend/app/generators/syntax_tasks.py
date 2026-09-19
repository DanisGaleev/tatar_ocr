import random
from typing import Optional, List, Dict, Any
from app.generators.base import BaseTaskGenerator, TaskDraft
from app.generators.lexicon import (
    SENTENCE_CLOZE_SETS,
    TRUE_FALSE_SETS,
    FIND_ERROR_SETS,
    WORD_ORDER_SETS,
)


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

        opts_str = f"А) {opts[0]}   Б) {opts[1]}   В) {opts[2]}"
        prompt = f"«{sent}»\nДөрес кушымчаны сайлагыз: {opts_str}"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag="sentence_cloze",
            topic_name_tt="Җөмләдә кушымча",
            prompt_tt=prompt,
            expected_answer=ans.upper(),
            cell_count=1,
            grade_level=grade_level,
            metadata={"sentence": sent, "options": opts},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = list(SENTENCE_CLOZE_SETS)
        rng.shuffle(pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item = pool[i % len(pool)]
            sent, opts, ans = item
            opts_str = f"А) {opts[0]}   Б) {opts[1]}   В) {opts[2]}"
            draft = TaskDraft(
                task_type=self.task_type,
                topic_tag="sentence_cloze",
                topic_name_tt="Җөмләдә кушымча",
                prompt_tt=f"«{sent}»\nДөрес кушымчаны сайлагыз: {opts_str}",
                expected_answer=ans.upper(),
                cell_count=1,
                grade_level=grade_level,
                metadata={"sentence": sent, "options": opts},
            )
            drafts.append(draft)
        return drafts

    def create_custom(self, sentence: str, options: List[str], correct_letter: str, grade_level: int = 7, **kwargs: Any) -> TaskDraft:
        opts_str = "   ".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options[:3]))
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="sentence_cloze",
            topic_name_tt="Җөмләдә кушымча",
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
            topic_tag="true_false",
            topic_name_tt="Дөрес яки ялгыш",
            prompt_tt=prompt,
            expected_answer=ans,
            cell_count=1,
            grade_level=grade_level,
            metadata={"statement": stmt, "is_true": is_true},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = list(TRUE_FALSE_SETS)
        rng.shuffle(pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item = pool[i % len(pool)]
            stmt, is_true = item
            ans = "Д" if is_true else "Я"
            draft = TaskDraft(
                task_type=self.task_type,
                topic_tag="true_false",
                topic_name_tt="Дөрес яки ялгыш",
                prompt_tt=f"{stmt} — Раслау дөресме? (Д — дөрес, Я — ялгыш)",
                expected_answer=ans,
                cell_count=1,
                grade_level=grade_level,
                metadata={"statement": stmt, "is_true": is_true},
            )
            drafts.append(draft)
        return drafts

    def create_custom(self, statement: str, is_true: bool, grade_level: int = 6, **kwargs: Any) -> TaskDraft:
        ans = "Д" if is_true else "Я"
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="true_false",
            topic_name_tt="Дөрес яки ялгыш",
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
        return "Хатаны тап (җөмлә санын)"

    def generate(
        self,
        seed: Optional[int] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> TaskDraft:
        rng = random.Random(seed) if seed is not None else random.Random()
        item = rng.choice(FIND_ERROR_SETS)
        sentences, err_num = item

        sents_formatted = "   ".join(sentences)
        prompt = f"Кайсы җөмләдә грамматик хата бар? (1-4 языгыз): {sents_formatted}"

        return TaskDraft(
            task_type=self.task_type,
            topic_tag="find_error",
            topic_name_tt="Хатаны тап",
            prompt_tt=prompt,
            expected_answer=str(err_num),
            cell_count=1,
            grade_level=grade_level,
            metadata={"sentences": sentences},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        grade_level: int = 7,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = list(FIND_ERROR_SETS)
        rng.shuffle(pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item = pool[i % len(pool)]
            sentences, err_num = item
            sents_formatted = "   ".join(sentences)
            draft = TaskDraft(
                task_type=self.task_type,
                topic_tag="find_error",
                topic_name_tt="Хатаны тап",
                prompt_tt=f"Кайсы җөмләдә грамматик хата бар? (1-4 языгыз): {sents_formatted}",
                expected_answer=str(err_num),
                cell_count=1,
                grade_level=grade_level,
                metadata={"sentences": sentences},
            )
            drafts.append(draft)
        return drafts

    def create_custom(self, sentences: List[str], error_number: int, grade_level: int = 7, **kwargs: Any) -> TaskDraft:
        sents_formatted = "   ".join(sentences)
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="find_error",
            topic_name_tt="Хатаны тап",
            prompt_tt=f"Кайсы җөмләдә грамматик хата бар? (1-4 языгыз): {sents_formatted}",
            expected_answer=str(error_number),
            cell_count=1,
            grade_level=grade_level,
            metadata={"sentences": sentences, "is_custom": True},
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
            topic_tag="word_order",
            topic_name_tt="Сүзләр тәртибе",
            prompt_tt=prompt,
            expected_answer=ans_order.upper(),
            cell_count=len(ans_order),
            grade_level=grade_level,
            metadata={"parts": parts},
        )

    def generate_batch(
        self,
        count: int,
        seed: Optional[int] = None,
        grade_level: int = 6,
        **kwargs: Any,
    ) -> List[TaskDraft]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pool = list(WORD_ORDER_SETS)
        rng.shuffle(pool)
        drafts: List[TaskDraft] = []
        for i in range(count):
            item = pool[i % len(pool)]
            parts, ans_order = item
            parts_formatted = "   ".join(parts)
            draft = TaskDraft(
                task_type=self.task_type,
                topic_tag="word_order",
                topic_name_tt="Сүзләр тәртибе",
                prompt_tt=f"Җөмлә төзегез (хәрефләр тәртибен языгыз): {parts_formatted}",
                expected_answer=ans_order.upper(),
                cell_count=len(ans_order),
                grade_level=grade_level,
                metadata={"parts": parts},
            )
            drafts.append(draft)
        return drafts

    def create_custom(self, parts: List[str], correct_order: str, grade_level: int = 6, **kwargs: Any) -> TaskDraft:
        parts_formatted = "   ".join(parts)
        clean_order = correct_order.strip().upper()
        return TaskDraft(
            task_type=self.task_type,
            topic_tag="word_order",
            topic_name_tt="Сүзләр тәртибе",
            prompt_tt=f"Җөмлә төзегез (хәрефләр тәртибен языгыз): {parts_formatted}",
            expected_answer=clean_order,
            cell_count=len(clean_order),
            grade_level=grade_level,
            metadata={"parts": parts, "is_custom": True},
        )
