from functools import lru_cache

from deep_translator import GoogleTranslator
from langdetect import LangDetectException, detect

from summarizer.config import LENGTH_PRESETS, MIN_INPUT_CHARS, MODEL_NAME


class SummarizationError(ValueError):
    pass


@lru_cache(maxsize=1)
def _load_model():
    from transformers import BartForConditionalGeneration, BartTokenizer

    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model


def get_summarizer():
    tokenizer, model = _load_model()

    def _summarize(text, *, min_length, max_length, do_sample=False):
        inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
        generated_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            min_length=min_length,
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        summary_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return [{"summary_text": summary_text}]

    return _summarize


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
