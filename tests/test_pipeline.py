import pytest

from summarizer.pipeline import SummarizationError, summarize


class FakeSummarizer:
    def __call__(self, text, **kwargs):
        return [{"summary_text": "This is a fake summary."}]


def fake_translator(text, source, target):
    return f"[{target}] {text}"


def test_rejects_short_input():
    with pytest.raises(SummarizationError):
        summarize("too short")


def test_rejects_unknown_length_preset():
    long_text = "word " * 20
    with pytest.raises(SummarizationError):
        summarize(long_text, length="extra-long")


def test_skips_translation_for_english_input():
    text = "This is a sufficiently long piece of English text for testing. " * 2
    calls = []

    def tracking_translator(t, s, tgt):
        calls.append((s, tgt))
        return fake_translator(t, s, tgt)

    result = summarize(
        text,
        output_language="en",
        summarizer=FakeSummarizer(),
        translator=tracking_translator,
    )

    assert calls == []
    assert result["summary"] == "This is a fake summary."
    assert result["detected_language"] == "en"


def test_translates_non_english_input_and_output():
    text = "Ceci est un texte suffisamment long pour être résumé correctement. " * 2
    result = summarize(
        text,
        output_language="fr",
        summarizer=FakeSummarizer(),
        translator=fake_translator,
    )

    assert result["summary"] == "[fr] This is a fake summary."


def test_computes_compression_ratio():
    text = "word " * 100
    result = summarize(text, summarizer=FakeSummarizer(), translator=fake_translator)

    assert 0 <= result["compression_ratio"] <= 1
    assert result["original_word_count"] == 100
