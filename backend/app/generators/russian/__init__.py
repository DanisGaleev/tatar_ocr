from app.generators.russian.phonetics import (
    RUSSIAN_UPPERCASE_33,
    TATAR_ONLY_LETTERS,
    clean_russian_word,
    mask_orthogram_combination,
    analyze_russian_phonetics,
)
from app.generators.russian.orthography_tasks import (
    RuOrthogramCombinationsGenerator,
    RuUnstressedVowelGenerator,
    RuPairedConsonantGenerator,
    RuUnpronounceableConsonantGenerator,
    RuSoftHardSignGenerator,
    RuDictionaryWordsGenerator,
)
from app.generators.russian.morphology_tasks import (
    RuNounGenderGenerator,
    RuNounNumberGenerator,
    RuPartsOfSpeechGenerator,
    RuNounCaseGenerator,
)
from app.generators.russian.lexicon_tasks import (
    RuAntonymsGenerator,
    RuSynonymsGenerator,
)
from app.generators.russian.phonetics_tasks import (
    RuPhoneticsGenerator,
    RuOddOneOutGenerator,
)

__all__ = [
    "RUSSIAN_UPPERCASE_33",
    "TATAR_ONLY_LETTERS",
    "clean_russian_word",
    "mask_orthogram_combination",
    "analyze_russian_phonetics",
    "RuOrthogramCombinationsGenerator",
    "RuUnstressedVowelGenerator",
    "RuPairedConsonantGenerator",
    "RuUnpronounceableConsonantGenerator",
    "RuSoftHardSignGenerator",
    "RuDictionaryWordsGenerator",
    "RuNounGenderGenerator",
    "RuNounNumberGenerator",
    "RuPartsOfSpeechGenerator",
    "RuNounCaseGenerator",
    "RuAntonymsGenerator",
    "RuSynonymsGenerator",
    "RuPhoneticsGenerator",
    "RuOddOneOutGenerator",
]
