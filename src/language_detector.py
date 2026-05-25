from langdetect import detect, LangDetectException
from langdetect import DetectorFactory

DetectorFactory.seed = 0  # make detection deterministic across reruns

_INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, Sanskrit)
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]

_INDIC_LANG_CODES = {"hi", "ta", "te", "kn", "ml", "bn", "gu", "pa", "mr", "ur"}


def _has_indic_script(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for start, end in _INDIC_RANGES:
            if start <= cp <= end:
                return True
    return False


def detect_language(text: str) -> str:
    """Return 'indic' or 'english'.

    Stage 1: any Indic Unicode character → 'indic' immediately (covers native-script
    queries in Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati,
    Gurmukhi). Stage 2: langdetect on Latin-script text for Hinglish / Tanglish.
    Falls through to 'english' if detection is uncertain — GPT-4o handles
    misclassified Latin-script Indic queries correctly (see Decision #27).
    """
    if _has_indic_script(text):
        return "indic"

    try:
        lang = detect(text)
        if lang in _INDIC_LANG_CODES:
            return "indic"
    except LangDetectException:
        pass

    return "english"
