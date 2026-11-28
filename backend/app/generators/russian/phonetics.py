import re
from typing import Dict, Any, Tuple
from app.generators.phonetics import PhoneticsError

RUSSIAN_UPPERCASE_33 = tuple("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
TATAR_ONLY_LETTERS = set("ӘӨҮҖҢҺәөүҗңһ")
RUSSIAN_CYRILLIC_PATTERN = re.compile(r"^[А-Яа-яЁё\- ]+$")

RU_VOWELS = set("аеёиоуыэюя")
RU_IOTATED_VOWELS = set("еёюя")
RU_SOFTENING_VOWELS = set("еёиюя")
RU_ALWAYS_HARD = set("жшц")
RU_ALWAYS_SOFT = set("чщй")
RU_CONSONANTS = set("бвгджзйклмнпрстфхцчшщ")
RU_SILENT_SIGNS = set("ъь")

ORTHOGRAM_COMBINATIONS = ("жи", "ши", "ча", "ща", "чу", "щу")


def clean_russian_word(word: str) -> str:
    """Validate and trim a Russian Cyrillic word (rejects Tatar-only letters, Latin, digits)."""
    cleaned = word.strip()
    if not cleaned:
        raise PhoneticsError("Word cannot be empty or whitespace only.")
    if any(ch in TATAR_ONLY_LETTERS for ch in cleaned):
        raise PhoneticsError(f"Word '{word}' contains Tatar-specific letters not allowed in Russian mode.")
    if not RUSSIAN_CYRILLIC_PATTERN.match(cleaned):
        raise PhoneticsError(f"Word '{word}' contains non-Russian or invalid characters.")
    return cleaned


def mask_orthogram_combination(word: str) -> Tuple[str, str]:
    """
    Finds the first primary-school orthogram combination (жи/ши/ча/ща/чу/щу) in a Russian word,
    replaces its vowel with '_', and returns (masked_word, rule_label).
    """
    cleaned = clean_russian_word(word).lower()
    if " " in cleaned or "-" in cleaned:
        raise PhoneticsError(f"Word '{word}' must be a single word without spaces or hyphens.")
    for combo in ORTHOGRAM_COMBINATIONS:
        idx = cleaned.find(combo)
        if idx != -1:
            masked = cleaned[: idx + 1] + "_" + cleaned[idx + 2 :]
            if combo in ("жи", "ши"):
                rule = "ЖИ–ШИ"
            elif combo in ("ча", "ща"):
                rule = "ЧА–ЩА"
            else:
                rule = "ЧУ–ЩУ"
            return masked, rule
    raise PhoneticsError(f"Word '{word}' does not contain ЖИ/ШИ, ЧА/ЩА, or ЧУ/ЩУ.")


def analyze_russian_phonetics(word: str) -> Dict[str, Any]:
    """
    Deterministic Russian primary-school phonetic analysis:
      - letter_count: total letters
      - sound_count: 'ъ'/'ь' give 0 sounds; iotated 'е/ё/ю/я' give 2 sounds
        at word start, after any vowel, or after 'ъ'/'ь'
      - all_consonants_hard: True if every consonant is hard (no always-soft,
        no soft-following vowel or 'ь' after a non-always-hard consonant)
      - all_consonants_soft: True if every consonant is soft (either always-soft
        or followed by softening vowel/'ь', and no always-hard consonant)
    """
    cleaned = clean_russian_word(word).lower()
    letters = [ch for ch in cleaned if ch.isalpha()]
    letter_count = len(letters)

    sound_count = 0
    consonant_states = []

    for i, ch in enumerate(letters):
        if ch in RU_SILENT_SIGNS:
            continue
        if ch in RU_IOTATED_VOWELS:
            prev = letters[i - 1] if i > 0 else None
            if prev is None or prev in RU_VOWELS or prev in RU_SILENT_SIGNS:
                sound_count += 2
                # Iotated vowel in strong position produces soft consonant sound [й']
                consonant_states.append("soft")
            else:
                sound_count += 1
        else:
            sound_count += 1

        if ch in RU_CONSONANTS:
            if ch in RU_ALWAYS_HARD:
                consonant_states.append("hard")
            elif ch in RU_ALWAYS_SOFT:
                consonant_states.append("soft")
            else:
                nxt = letters[i + 1] if i + 1 < letter_count else None
                if nxt is not None and (nxt in RU_SOFTENING_VOWELS or nxt == "ь"):
                    consonant_states.append("soft")
                else:
                    consonant_states.append("hard")

    return {
        "word": cleaned,
        "letter_count": letter_count,
        "sound_count": sound_count,
        "has_consonants": bool(consonant_states),
        "all_consonants_hard": bool(consonant_states) and all(s == "hard" for s in consonant_states),
        "all_consonants_soft": bool(consonant_states) and all(s == "soft" for s in consonant_states),
    }
