from functools import lru_cache

from deep_translator import GoogleTranslator
from langdetect import LangDetectException, detect

from summarizer.config import LENGTH_PRESETS, MIN_INPUT_CHARS, MODEL_NAME


class SummarizationError(ValueError):
    pass


@lru_cache(maxsize=1)
def get_summarizer():
    from transformers import pipeline

    return pipeline("summarization", model=MODEL_NAME)


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def translate(text: str, source: str, target: str) -> str:
    if source == target:
        return text
    return GoogleTranslator(source=source, target=target).translate(text)


def summarize(
    text: str,
    *,
    output_language: str = "en",
    length: str = "medium",
    summarizer=None,
    translator=translate,
) -> dict:
    if not text or len(text.strip()) < MIN_INPUT_CHARS:
        raise SummarizationError(
            f"Input text must contain at least {MIN_INPUT_CHARS} characters."
        )

    if length not in LENGTH_PRESETS:
        raise SummarizationError(f"Unknown length preset: {length!r}")

    preset = LENGTH_PRESETS[length]
    source_language = detect_language(text)

    text_en = translator(text, source_language, "en") if source_language != "en" else text

    summarizer_fn = summarizer or get_summarizer()
    generated = summarizer_fn(
        text_en,
        min_length=preset["min_length"],
        max_length=preset["max_length"],
        do_sample=False,
    )
    summary_en = generated[0]["summary_text"]

    summary = (
        translator(summary_en, "en", output_language)
        if output_language != "en"
        else summary_en
    )

    original_words = len(text.split())
    summary_words = len(summary.split())
    compression_ratio = 1 - (summary_words / original_words) if original_words else 0.0

    return {
        "summary": summary,
        "detected_language": source_language,
        "original_word_count": original_words,
        "summary_word_count": summary_words,
        "compression_ratio": compression_ratio,
    }
