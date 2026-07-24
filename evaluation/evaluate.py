import json
from pathlib import Path

import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer

from summarizer.pipeline import summarize, translate

DATA_PATH = Path(__file__).parent / "test_texts.json"
REPORT_PATH = Path(__file__).parent / "report.png"


def load_test_texts() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def evaluate() -> pd.DataFrame:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []

    for lang, text in load_test_texts().items():
        text_en = translate(text, lang, "en") if lang != "en" else text
        result = summarize(text, output_language="en", length="medium")
        summary_en = result["summary"]

        rouge = scorer.score(text_en, summary_en)
        _, _, f1 = bert_score([summary_en], [text_en], lang="en", verbose=False)

        rows.append(
            {
                "language": lang.upper(),
                "original_words": result["original_word_count"],
                "summary_words": result["summary_word_count"],
                "compression": result["compression_ratio"] * 100,
                "rouge1_f1": rouge["rouge1"].fmeasure,
                "rouge2_f1": rouge["rouge2"].fmeasure,
                "rougeL_f1": rouge["rougeL"].fmeasure,
                "bertscore_f1": f1.item(),
            }
        )

    return pd.DataFrame(rows)


def plot_report(df: pd.DataFrame, output_path: Path = REPORT_PATH) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df.set_index("language")[["rouge1_f1", "rouge2_f1", "rougeL_f1"]].plot(
        kind="bar", ax=axes[0], edgecolor="black"
    )
    axes[0].set_title("ROUGE F1 by language")
    axes[0].set_ylabel("F1 score")

    df.set_index("language")["bertscore_f1"].plot(
        kind="bar", ax=axes[1], color="#16a085", edgecolor="black"
    )
    axes[1].set_title("BERTScore F1 by language")
    axes[1].set_ylim(0.8, 1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


if __name__ == "__main__":
    results = evaluate()
    print(results.to_string(index=False))
    plot_report(results)
    print(f"\nSaved chart to {REPORT_PATH}")
