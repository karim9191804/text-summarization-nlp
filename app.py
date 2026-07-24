import gradio as gr

from summarizer.config import LENGTH_PRESETS, SUPPORTED_OUTPUT_LANGUAGES
from summarizer.file_io import UnsupportedFileTypeError, read_text_from_file
from summarizer.pipeline import SummarizationError, summarize

LENGTH_CHOICES = [preset["label"] for preset in LENGTH_PRESETS.values()]
LABEL_TO_KEY = {preset["label"]: key for key, preset in LENGTH_PRESETS.items()}


def run(text_input, file_input, output_lang_label, length_label):
    if file_input is not None:
        try:
            text_input = read_text_from_file(file_input.name)
        except UnsupportedFileTypeError as exc:
            return f"❌ {exc}"

    try:
        result = summarize(
            text_input,
            output_language=SUPPORTED_OUTPUT_LANGUAGES[output_lang_label],
            length=LABEL_TO_KEY[length_label],
        )
    except SummarizationError as exc:
        return f"❌ {exc}"

    stats = (
        f"\n\n---\n"
        f"**{result['original_word_count']}** words → "
        f"**{result['summary_word_count']}** words "
        f"({result['compression_ratio']:.0%} compression) · "
        f"detected input language: `{result['detected_language']}`"
    )
    return result["summary"] + stats


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 📝 Multilingual Text Summarizer
        Abstractive summarization powered by BART, wrapped with automatic
        language detection and translation for 100+ input languages.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text to summarize",
                placeholder="Paste your text here (minimum 50 characters)...",
                lines=10,
            )
            file_input = gr.File(
                label="...or upload a file",
                file_types=[".txt", ".md", ".pdf", ".docx"],
            )

        with gr.Column(scale=1):
            output_lang = gr.Dropdown(
                choices=list(SUPPORTED_OUTPUT_LANGUAGES.keys()),
                value="English",
                label="Output language",
            )
            length_choice = gr.Radio(
                choices=LENGTH_CHOICES,
                value=LENGTH_PRESETS["medium"]["label"],
                label="Summary length",
            )
            submit_btn = gr.Button("Generate summary", variant="primary")

    output = gr.Textbox(label="Summary", lines=8)

    submit_btn.click(
        fn=run,
        inputs=[text_input, file_input, output_lang, length_choice],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch(title="Multilingual Text Summarizer", theme=gr.themes.Soft())
