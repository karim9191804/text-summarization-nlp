import pytest

from summarizer.file_io import UnsupportedFileTypeError, read_text_from_file


def test_reads_plain_text_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world", encoding="utf-8")

    assert read_text_from_file(file_path) == "hello world"


def test_reads_markdown_file(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\n\nBody text.", encoding="utf-8")

    assert read_text_from_file(file_path) == "# Title\n\nBody text."


def test_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "sample.exe"
    file_path.write_text("binary-ish", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        read_text_from_file(file_path)
