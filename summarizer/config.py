MODEL_NAME = "karimhoucem/Multilingual_Text_Summarization_System-BART_v1.0.9"

MIN_INPUT_CHARS = 50

LENGTH_PRESETS = {
    "short": {"min_length": 60, "max_length": 100, "label": "Short (~80 words)"},
    "medium": {"min_length": 120, "max_length": 180, "label": "Medium (~150 words)"},
    "long": {"min_length": 220, "max_length": 300, "label": "Long (~250 words)"},
}

SUPPORTED_OUTPUT_LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Arabic": "ar",
}
