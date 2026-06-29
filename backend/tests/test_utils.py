"""Unit tests for backend utility functions."""

import os
import io
import tempfile

import pandas as pd
import pytest

from backend.utils import (
    get_dataset_path,
    load_dataset,
    save_dataset,
    get_column_types,
    format_file_size,
    validate_csv,
    UPLOAD_FOLDER,
)


class TestGetDatasetPath:
    def test_returns_full_path(self):
        result = get_dataset_path("test.csv")
        assert result == os.path.join(UPLOAD_FOLDER, "test.csv")

    def test_empty_filename_raises(self):
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            get_dataset_path("")

    def test_none_filename_raises(self):
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            get_dataset_path(None)


class TestLoadDataset:
    def test_loads_existing_csv(self, tmp_path, monkeypatch):
        # Create a temp CSV in a fake uploads dir
        monkeypatch.setattr("backend.utils.UPLOAD_FOLDER", str(tmp_path))
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv(tmp_path / "test.csv", index=False)

        result = load_dataset("test.csv")
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_file_not_found_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.utils.UPLOAD_FOLDER", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_dataset("nonexistent.csv")

    def test_empty_filename_raises(self):
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            load_dataset("")


class TestSaveDataset:
    def test_saves_dataframe(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.utils.UPLOAD_FOLDER", str(tmp_path))
        df = pd.DataFrame({"x": [10, 20], "y": ["a", "b"]})

        path = save_dataset(df, "output.csv")
        assert os.path.isfile(path)

        loaded = pd.read_csv(path)
        assert list(loaded.columns) == ["x", "y"]
        assert len(loaded) == 2

    def test_non_dataframe_raises(self):
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            save_dataset([1, 2, 3], "test.csv")

    def test_empty_filename_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            save_dataset(df, "")


class TestGetColumnTypes:
    def test_mixed_types(self):
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "num2": [4, 5, 6],
            "cat1": ["a", "b", "c"],
            "cat2": pd.Categorical(["x", "y", "z"]),
        })
        result = get_column_types(df)
        assert "num1" in result["numerical"]
        assert "num2" in result["numerical"]
        assert "cat1" in result["categorical"]
        assert "cat2" in result["categorical"]

    def test_all_numerical(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        result = get_column_types(df)
        assert len(result["numerical"]) == 2
        assert len(result["categorical"]) == 0

    def test_all_categorical(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["m", "n"]})
        result = get_column_types(df)
        assert len(result["numerical"]) == 0
        assert len(result["categorical"]) == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = get_column_types(df)
        assert result == {"numerical": [], "categorical": []}

    def test_non_dataframe_raises(self):
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            get_column_types({"a": [1, 2]})


class TestFormatFileSize:
    def test_zero_bytes(self):
        assert format_file_size(0) == "0 bytes"

    def test_small_bytes(self):
        assert format_file_size(512) == "512 bytes"

    def test_one_byte(self):
        assert format_file_size(1) == "1 bytes"

    def test_just_below_kb(self):
        assert format_file_size(1023) == "1023 bytes"

    def test_exactly_1kb(self):
        assert format_file_size(1024) == "1.00 KB"

    def test_kilobytes(self):
        assert format_file_size(1536) == "1.50 KB"

    def test_just_below_mb(self):
        result = format_file_size(1024 * 1024 - 1)
        assert "KB" in result

    def test_exactly_1mb(self):
        assert format_file_size(1024 * 1024) == "1.00 MB"

    def test_megabytes(self):
        assert format_file_size(2 * 1024 * 1024 + 512 * 1024) == "2.50 MB"

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            format_file_size(-1)


class FakeFileStorage:
    """Mock Flask FileStorage for testing validate_csv."""

    def __init__(self, filename=None, content_length=None, data=b""):
        self.filename = filename
        self.content_length = content_length
        self.stream = io.BytesIO(data)


class TestValidateCsv:
    def test_valid_csv(self):
        file = FakeFileStorage(filename="data.csv", content_length=1000)
        is_valid, error = validate_csv(file)
        assert is_valid is True
        assert error is None

    def test_valid_csv_uppercase_extension(self):
        file = FakeFileStorage(filename="Data.CSV", content_length=500)
        is_valid, error = validate_csv(file)
        assert is_valid is True
        assert error is None

    def test_valid_csv_mixed_case(self):
        file = FakeFileStorage(filename="report.Csv", content_length=100)
        is_valid, error = validate_csv(file)
        assert is_valid is True
        assert error is None

    def test_none_file(self):
        is_valid, error = validate_csv(None)
        assert is_valid is False
        assert "No file provided" in error

    def test_no_filename(self):
        file = FakeFileStorage(filename="", content_length=100)
        is_valid, error = validate_csv(file)
        assert is_valid is False
        assert "No filename" in error

    def test_wrong_extension(self):
        file = FakeFileStorage(filename="data.xlsx", content_length=100)
        is_valid, error = validate_csv(file)
        assert is_valid is False
        assert "CSV" in error

    def test_empty_file(self):
        file = FakeFileStorage(filename="empty.csv", content_length=0, data=b"")
        is_valid, error = validate_csv(file)
        assert is_valid is False
        assert "empty" in error.lower()

    def test_file_too_large(self):
        file = FakeFileStorage(filename="big.csv", content_length=51 * 1024 * 1024)
        is_valid, error = validate_csv(file)
        assert is_valid is False
        assert "50 MB" in error

    def test_exactly_max_size(self):
        file = FakeFileStorage(filename="max.csv", content_length=50 * 1024 * 1024)
        is_valid, error = validate_csv(file)
        assert is_valid is True
        assert error is None

    def test_exactly_1_byte(self):
        file = FakeFileStorage(filename="tiny.csv", content_length=1)
        is_valid, error = validate_csv(file)
        assert is_valid is True
        assert error is None

    def test_size_from_stream_fallback(self):
        data = b"col1,col2\n1,2\n3,4\n"
        file = FakeFileStorage(filename="data.csv", content_length=None, data=data)
        is_valid, error = validate_csv(file)
        assert is_valid is True
        assert error is None
