import re
from typing import List, Dict, Any, Optional, Tuple

FRONT_VOWELS = set("әөүеиэӘӨҮЕИЭ")
BACK_VOWELS = set("аоуыАОУЫ")
ALL_VOWELS = FRONT_VOWELS | BACK_VOWELS | set("яюЯЮ")

VOICELESS_CONSONANTS = set("пктсфхчщшцПКТСФХЧЩШЦ")
NASAL_CONSONANTS = set("мнңМНҢ")

TATAR_CYRILLIC_PATTERN = re.compile(r"^[А-Яа-яЁёӘәӨөҮүҖҗҢңҺһ\- ]+$")

CASE_NAMES = {
    "case_genitive": "Иялек",
    "case_dative": "Юнәлеш",
    "case_accusative": "Төшем",
    "case_locative": "Урын-вакыт",
    "case_ablative": "Чыгыш",
}

class PhoneticsError(ValueError):
    """Raised when an invalid word stem is provided for phonetic inflection."""
    pass

def clean_word(word: str) -> str:
    cleaned = word.strip()
    if not cleaned:
        raise PhoneticsError("Word cannot be empty or whitespace only.")
    if not TATAR_CYRILLIC_PATTERN.match(cleaned):
        raise PhoneticsError(f"Word '{word}' contains non-Tatar or invalid characters.")
    return cleaned

def get_last_vowel(word: str) -> str:
    cleaned = clean_word(word).lower()
    for char in reversed(cleaned):
        if char in ALL_VOWELS:
            return char
    raise PhoneticsError(f"Word '{word}' contains no detectable vowels.")

def is_front_vowel(word: str) -> bool:
    """
    Returns True if the word harmony is front (нечкә), False if back (калын).
    Adheres strictly to Tatar phonological rules:
      - Words with unambiguous Tatar vowels (ә, ө, ү vs о, у, ы)
      - Handles loanwords and diphthongs where 'е' or 'и' follow back vowels (e.g. каен -> калын, тарих -> калын).
    """
    cleaned = clean_word(word).lower()
    
    # Handle Arabic loanwords where 'гый' precedes front vowels (e.g. гыйлем -> front)
    if cleaned.startswith("гый") and any(c in FRONT_VOWELS for c in cleaned):
        return True

    # Check for unambiguous Tatar-specific front/back vowels
    has_specific_front = any(c in "әөү" for c in cleaned)
    has_specific_back = any(c in "оуы" for c in cleaned)
    
    if has_specific_front and not has_specific_back:
        return True
    if has_specific_back and not has_specific_front:
        return False

    vowels = [c for c in cleaned if c in ALL_VOWELS]
    if not vowels:
        raise PhoneticsError(f"Word '{word}' contains no detectable vowels.")

    last_v = vowels[-1]

    # In Tatar orthography, 'е' preceded by back vowels represents [йы] / [йɤ] (e.g. каен -> калын)
    if last_v in "е":
        if any(c in BACK_VOWELS for c in vowels[:-1]):
            return False
        return True

    # Similarly, 'и' preceded by back vowels (e.g. тарих, шагыйрь) follows back harmony
    if last_v in "и":
        if any(c in BACK_VOWELS for c in vowels[:-1]):
            return False
        return True

    if cleaned.endswith("ь"):
        return True

    if last_v in FRONT_VOWELS:
        return True
    if last_v in BACK_VOWELS:
        return False

    return False


def get_case_suffix(stem: str, case_code: str) -> str:
    """
    Computes standard Tatar case suffix adhering to vowel harmony and consonant assimilation.
    Correctly handles words ending in soft sign 'ь' (palatalized preceding consonant).
    
    Cases:
      - case_genitive (Иялек): -ның / -нең
      - case_dative (Юнәлеш): -ка / -кә (after voiceless), -га / -гә (others)
      - case_accusative (Төшем): -ны / -не
      - case_locative (Урын-вакыт): -та / -тә (after voiceless), -да / -дә (others)
      - case_ablative (Чыгыш): -нан / -нән (after nasal), -тан / -тән (after voiceless), -дан / -дән (others)
    """
    cleaned = clean_word(stem).lower()
    if cleaned.endswith("ь") and len(cleaned) > 1:
        last_consonant = cleaned[-2]
        is_front = True
    else:
        last_consonant = cleaned[-1]
        is_front = is_front_vowel(cleaned)

    is_voiceless = last_consonant in VOICELESS_CONSONANTS
    is_nasal = last_consonant in NASAL_CONSONANTS

    if case_code == "case_genitive":
        return "нең" if is_front else "ның"
    
    elif case_code == "case_dative":
        if is_voiceless:
            return "кә" if is_front else "ка"
        return "гә" if is_front else "га"
        
    elif case_code == "case_accusative":
        return "не" if is_front else "ны"
        
    elif case_code == "case_locative":
        if is_voiceless:
            return "тә" if is_front else "та"
        return "дә" if is_front else "да"
        
    elif case_code == "case_ablative":
        if is_nasal:
            return "нән" if is_front else "нан"
        if is_voiceless:
            return "тән" if is_front else "тан"
        return "дән" if is_front else "дан"
        
    else:
        raise PhoneticsError(f"Unsupported case code: '{case_code}'")

def inflect_case(stem: str, case_code: str) -> str:
    """Inflects stem into given case and returns uppercase result."""
    suffix = get_case_suffix(stem, case_code)
    return (clean_word(stem) + suffix).upper()

def get_plural_suffix(stem: str) -> str:
    """
    Computes Tatar plural suffix:
      - after nasal consonants (м, н, ң): -нар / -нәр
      - after all other consonants and vowels: -лар / -ләр
      - words ending in soft sign 'ь' take front suffix (-ләр)
    """
    cleaned = clean_word(stem).lower()
    if cleaned.endswith("ь") and len(cleaned) > 1:
        last_consonant = cleaned[-2]
        is_front = True
    else:
        last_consonant = cleaned[-1]
        is_front = is_front_vowel(cleaned)

    is_nasal = last_consonant in NASAL_CONSONANTS

    if is_nasal:
        return "нәр" if is_front else "нар"
    return "ләр" if is_front else "лар"


def inflect_plural(stem: str) -> str:
    """Inflects stem into plural form and returns uppercase result."""
    suffix = get_plural_suffix(stem)
    return (clean_word(stem) + suffix).upper()

def build_expected_cells(word: str, cell_count: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Builds the exact cell-by-cell structure for the OCR verification offline bundle.
    Upper-cases letters, determines Unicode code point (e.g. U+041A),
    and pads remaining cells with empty space if cell_count is specified.
    """
    upper_word = word.strip().upper()
    total_cells = cell_count if cell_count and cell_count >= len(upper_word) else len(upper_word)
    
    cells = []
    for idx in range(total_cells):
        if idx < len(upper_word):
            ch = upper_word[idx]
            unicode_hex = f"U+{ord(ch):04X}"
            cells.append({
                "index": idx,
                "char": ch,
                "unicode": unicode_hex,
                "is_empty_allowed": False
            })
        else:
            cells.append({
                "index": idx,
                "char": " ",
                "unicode": "U+0020",
                "is_empty_allowed": True
            })
    return cells


def voice_stem_final_consonant(stem: str) -> str:
    """
    Applies Tatar consonant voicing (п -> б, к -> г) on morpheme boundary
    when followed by a vowel-initial suffix.
    """
    cleaned = clean_word(stem).lower()
    if cleaned.endswith("п"):
        return cleaned[:-1] + "б"
    if cleaned.endswith("к"):
        return cleaned[:-1] + "г"
    return cleaned


def get_possessive_suffix(stem: str, person: int = 1, number: str = "sg") -> Tuple[str, str]:
    """
    Returns (voiced_stem, suffix) for Tatar possessive category.
    person: 1 (I зат: минем), 2 (II зат: синең), 3 (III зат: аның)
    number: "sg" (берлек)
    """
    cleaned = clean_word(stem).lower()
    is_front = is_front_vowel(cleaned)
    last_char = cleaned[-1]
    ends_in_vowel = last_char in ALL_VOWELS

    # Special handling for monosyllabic 'су' and 'аю'
    if cleaned == "су":
        if person == 1:
            return "су", "ым"
        elif person == 2:
            return "су", "ың"
        elif person == 3:
            return "су", "ы"

    if cleaned == "аю":
        if person == 1:
            return "аю", "ым"
        elif person == 2:
            return "аю", "ың"
        elif person == 3:
            return "аю", "ы"

    # Special handling for stems ending in -ау / -әү (e.g. тау -> тавым/тавы, сорау -> соравым/соравы)
    if cleaned.endswith("ау") or cleaned.endswith("әү"):
        stem_v = cleaned[:-1] + "в"
        if person == 1:
            return stem_v, "ем" if is_front else "ым"
        elif person == 2:
            return stem_v, "ең" if is_front else "ың"
        elif person == 3:
            return stem_v, "е" if is_front else "ы"

    # Special handling for words ending in soft sign 'ь' (e.g. сәгать -> сәгатем, кәгазь -> кәгазем)
    if cleaned.endswith("ь") and len(cleaned) > 1:
        stem_no_soft = cleaned[:-1]
        if person == 1:
            return stem_no_soft, "ем" if is_front else "ым"
        elif person == 2:
            return stem_no_soft, "ең" if is_front else "ың"
        elif person == 3:
            return stem_no_soft, "е" if is_front else "ы"

    # Special handling for words ending in 'и' (e.g. әни, көри)
    if last_char == "и":
        if person == 1:
            return cleaned, "ем"
        elif person == 2:
            return cleaned, "ең"
        elif person == 3:
            return cleaned, "се"

    # Special handling for words ending in 'й' (e.g. өй, ай, чәй)
    # In Tatar orthography, 'й' drops before possessive suffix 'е': өй -> өем, өең, өе; ай -> аем, ае
    if last_char == "й":
        stem_without_y = cleaned[:-1]
        if person == 1:
            return stem_without_y, "ем"
        elif person == 2:
            return stem_without_y, "ең"
        elif person == 3:
            return stem_without_y, "е"

    if ends_in_vowel:
        if person == 1:
            return cleaned, "м"
        elif person == 2:
            return cleaned, "ң"
        elif person == 3:
            return cleaned, "се" if is_front else "сы"
    else:
        voiced = voice_stem_final_consonant(cleaned)
        if person == 1:
            return voiced, "ем" if is_front else "ым"
        elif person == 2:
            return voiced, "ең" if is_front else "ың"
        elif person == 3:
            return voiced, "е" if is_front else "ы"

    raise PhoneticsError(f"Unsupported person {person} or number {number}")


def inflect_possessive(stem: str, person: int = 1, number: str = "sg") -> str:
    """Inflects noun with possessive affix and returns uppercase result."""
    stem_part, suffix = get_possessive_suffix(stem, person, number)
    return (stem_part + suffix).upper()


def inflect_possessive_case(stem: str, person: int, case_code: str) -> str:
    """
    Inflects stem with possessive affix, then adds case affix.
    Handles the essential Tatar rules:
      - 3rd person possessive (-ы/-е, -сы/-се) inserts pronominal 'н' before ANY case affix:
        - Accusative: -ын / -ен, -сын / -сен (e.g. китабын, баласын)
        - Dative: -ына / -енә, -сына / -сенә (e.g. китабына, баласына)
        - Locative: -ында / -ендә, -сында / -сендә (e.g. китабында, баласында)
        - Ablative: -ыннан / -еннән, -сыннан / -сеннән (e.g. китабыннан, баласыннан)
        - Genitive: -ының / -енең, -сының / -сенең (e.g. китабының, баласының)
      - 1st/2nd person possessive:
        - Dative: -ма / -мә (1st: өемә, китабыма, балама), -ңа / -ңә (2nd: өеңә, китабыңа, балаңа)
        - Ablative: -мнан / -мнән (1st: өемнән, китабымнан), -ңнан / -ңнән (2nd: өеңнән, китабыңнан)
        - Locative: -мда / -мдә (1st: өемдә, китабымда), -ңда / -ңдә (2nd: өеңдә, китабыңда)
        - Accusative: -мне / -мны (1st: өемне, китабымны), -ңне / -ңны (2nd: өеңне, китабыңны)
        - Genitive: -мнең / -мның (1st: өемнең, китабымның), -ңнең / -ңның (2nd: өеңнең, китабыңның)
    """
    base_stem, poss_suffix = get_possessive_suffix(stem, person, "sg")
    combined_poss = base_stem + poss_suffix
    is_front = is_front_vowel(combined_poss)

    if person == 3:
        if case_code == "case_accusative":
            return (combined_poss + "н").upper()
        elif case_code == "case_dative":
            return (combined_poss + ("нә" if is_front else "на")).upper()
        elif case_code == "case_locative":
            return (combined_poss + ("ндә" if is_front else "нда")).upper()
        elif case_code == "case_ablative":
            return (combined_poss + ("ннән" if is_front else "ннан")).upper()
        elif case_code == "case_genitive":
            return (combined_poss + ("нең" if is_front else "ның")).upper()
        else:
            raise PhoneticsError(f"Unsupported case: {case_code}")
    else:
        # Person 1 or 2
        if case_code == "case_dative":
            return (combined_poss + ("ә" if is_front else "а")).upper()
        elif case_code == "case_ablative":
            return (combined_poss + ("нән" if is_front else "нан")).upper()
        elif case_code == "case_locative":
            return (combined_poss + ("дә" if is_front else "да")).upper()
        elif case_code == "case_accusative":
            return (combined_poss + ("не" if is_front else "ны")).upper()
        elif case_code == "case_genitive":
            return (combined_poss + ("нең" if is_front else "ның")).upper()
        else:
            raise PhoneticsError(f"Unsupported case: {case_code}")


def inflect_verb_negation(stem: str) -> str:
    """
    Present negative verb form: stem + -мый / -ми
    бар -> БАРМЫЙ, кил -> КИЛМИ, яз -> ЯЗМЫЙ, укы -> УКЫМЫЙ, эшлә -> ЭШЛӘМИ
    """
    cleaned = clean_word(stem).lower()
    is_front = is_front_vowel(cleaned)
    suffix = "ми" if is_front else "мый"
    return (cleaned + suffix).upper()


def inflect_verb_imperative(stem: str) -> str:
    """
    Imperative 2nd person plural:
      - after consonant: -ыгыз / -егез (with p->b voicing if applicable, e.g. тап -> табыгыз)
      - after vowel: -гыз / -гез (укы -> укыгыз, эшлә -> эшләгез)
    """
    cleaned = clean_word(stem).lower()
    is_front = is_front_vowel(cleaned)
    ends_in_vowel = cleaned[-1] in ALL_VOWELS

    if ends_in_vowel:
        suffix = "гез" if is_front else "гыз"
        return (cleaned + suffix).upper()
    else:
        voiced = voice_stem_final_consonant(cleaned)
        suffix = "егез" if is_front else "ыгыз"
        return (voiced + suffix).upper()


def inflect_comparative(stem: str) -> str:
    """
    Comparative degree of adjective: stem + -рак / -рәк
    зур -> ЗУРРАК, тиз -> ТИЗРӘК, салкын -> САЛКЫНРАК, яшел -> ЯШЕЛРӘК
    """
    cleaned = clean_word(stem).lower()
    is_front = is_front_vowel(cleaned)
    suffix = "рәк" if is_front else "рак"
    return (cleaned + suffix).upper()


def inflect_ordinal_numeral(stem: str) -> str:
    """
    Ordinal numerals:
      - after consonant: -ынчы / -енче (кырык -> кырыгынчы)
      - after vowel: -нчы / -нче (алты -> алтынчы, ике -> икенче)
    """
    cleaned = clean_word(stem).lower()
    is_front = is_front_vowel(cleaned)
    ends_in_vowel = cleaned[-1] in ALL_VOWELS

    if ends_in_vowel:
        suffix = "нче" if is_front else "нчы"
        return (cleaned + suffix).upper()
    else:
        voiced = voice_stem_final_consonant(cleaned)
        suffix = "енче" if is_front else "ынчы"
        return (voiced + suffix).upper()


def inflect_profession(stem: str) -> str:
    """
    Profession nouns (-чы / -че, or verbal base + -учы / -үче):
    балык -> БАЛЫКЧЫ, эш -> ЭШЧЕ, җыр -> ҖЫРЧЫ, укыту -> УКЫТУЧЫ, язу -> ЯЗУЧЫ
    """
    cleaned = clean_word(stem).lower()
    is_front = is_front_vowel(cleaned)
    suffix = "че" if is_front else "чы"
    return (cleaned + suffix).upper()


def inflect_abstract_noun(stem: str) -> str:
    """
    Abstract nouns (-лык / -лек):
    матур -> МАТУРЛЫК, иркен -> ИРКЕНЛЕК, дус -> ДУСЛЫК, яшь -> ЯШЬЛЕК
    """
    cleaned = clean_word(stem).lower()
    is_front = is_front_vowel(cleaned)
    suffix = "лек" if is_front else "лык"
    return (cleaned + suffix).upper()


def inflect_consonant_voicing(stem: str) -> str:
    """
    Illustrates consonant voicing (тартык алмашу) with 3rd person possessive:
    тарак + ы -> ТАРАГЫ, китап + ы -> КИТАБЫ, чиләк + е -> ЧИЛӘГЕ
    """
    cleaned = clean_word(stem).lower()
    if not (cleaned.endswith("п") or cleaned.endswith("к")):
        raise PhoneticsError(f"Stem '{stem}' does not end in voicable consonant (п, к).")
    return inflect_possessive(cleaned, person=3, number="sg")
