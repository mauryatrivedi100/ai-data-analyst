"""Property-based tests for upload validation and file size formatting."""

import io
import re

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.utils import validate_csv, format_file_size, MAX_FILE_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeFileStorage:
    """Mimics a Flask FileStorage object for testing validate_csv."""

    def __init__(self, filename, content_length):
        self.filename = filename
        self.content_length = content_length
        self.stream = io.BytesIO(b"x" * min(content_length, 16))


# Strategies
csv_extension = st.sampled_from([".csv", ".CSV", ".Csv", ".cSv", ".csV"])
non_csv_extension = st.sampled_from([
    ".xlsx", ".txt", ".json", ".parquet", ".tsv", ".xls", ".pdf", ".doc", ""
])
basename = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
)

# Size ranges
valid_size = st.integers(min_value=1, max_value=MAX_FILE_SIZE)          # 1 byte to 50 MB
too_small_size = st.just(0)                                              # 0 bytes (empty)
too_large_size = st.integers(min_value=MAX_FILE_SIZE + 1, max_value=MAX_FILE_SIZE * 3)


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 1: File Upload Validation
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(name=basename, ext=csv_extension, size=valid_size)
def test_valid_csv_files_are_accepted(name, ext, size):
    """A file with .csv extension (any case) and size 1–50 MB is accepted."""
    filename = name + ext
    file = FakeFileStorage(filename=filename, content_length=size)
    is_valid, error = validate_csv(file)
    assert is_valid is True, f"Expected valid for {filename!r} ({size} bytes), got error: {error}"
    assert error is None


@settings(max_examples=100)
@given(name=basename, ext=non_csv_extension, size=valid_size)
def test_non_csv_extension_is_rejected(name, ext, size):
    """A file without .csv extension is rejected regardless of size."""
    filename = name + ext
    # Ensure the filename doesn't accidentally end with .csv
    assume(not filename.lower().endswith(".csv"))
    file = FakeFileStorage(filename=filename, content_length=size)
    is_valid, error = validate_csv(file)
    assert is_valid is False, f"Expected rejection for {filename!r}"
    assert error is not None


@settings(max_examples=100)
@given(name=basename, ext=csv_extension, size=too_large_size)
def test_oversized_csv_is_rejected(name, ext, size):
    """A .csv file exceeding 50 MB is rejected."""
    filename = name + ext
    file = FakeFileStorage(filename=filename, content_length=size)
    is_valid, error = validate_csv(file)
    assert is_valid is False, f"Expected rejection for oversized file ({size} bytes)"
    assert error is not None


@settings(max_examples=100)
@given(name=basename, ext=csv_extension)
def test_empty_csv_is_rejected(name, ext):
    """A .csv file with 0 bytes (empty) is rejected."""
    filename = name + ext
    file = FakeFileStorage(filename=filename, content_length=0)
    # Stream is empty too
    file.stream = io.BytesIO(b"")
    is_valid, error = validate_csv(file)
    assert is_valid is False, f"Expected rejection for empty file {filename!r}"
    assert error is not None


@settings(max_examples=100)
@given(name=basename, ext=non_csv_extension, size=too_large_size)
def test_non_csv_and_oversized_is_rejected(name, ext, size):
    """A file that is both non-csv and oversized is still rejected."""
    filename = name + ext
    assume(not filename.lower().endswith(".csv"))
    file = FakeFileStorage(filename=filename, content_length=size)
    is_valid, error = validate_csv(file)
    assert is_valid is False


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 2: File Size Formatting
# Validates: Requirements 1.6
# ---------------------------------------------------------------------------

# Unit factors for parsing formatted output back to bytes
UNIT_FACTORS = {
    "bytes": 1,
    "KB": 1024,
    "MB": 1024 * 1024,
}

# Regex to parse formatted size strings like "512 bytes", "3.45 KB", "2.10 MB"
SIZE_PATTERN = re.compile(r"^([\d.]+)\s+(bytes|KB|MB)$")


@settings(max_examples=100)
@given(size=st.integers(min_value=0, max_value=MAX_FILE_SIZE * 2))
def test_format_file_size_round_trip(size):
    """Parsing the formatted string back yields the original within 1% tolerance."""
    formatted = format_file_size(size)

    match = SIZE_PATTERN.match(formatted)
    assert match is not None, f"Could not parse formatted string: {formatted!r}"

    numeric_str, unit = match.groups()
    numeric_value = float(numeric_str)
    factor = UNIT_FACTORS[unit]
    reconstructed = numeric_value * factor

    # For 0 bytes, exact match
    if size == 0:
        assert reconstructed == 0
    else:
        # Within 1% tolerance
        tolerance = size * 0.01
        assert abs(reconstructed - size) <= tolerance, (
            f"Round-trip failed: {size} bytes → {formatted!r} → {reconstructed} "
            f"(tolerance={tolerance})"
        )


@settings(max_examples=100)
@given(size=st.integers(min_value=0, max_value=MAX_FILE_SIZE * 2))
def test_format_file_size_correct_unit_selection(size):
    """Verify the correct unit is selected based on size thresholds."""
    formatted = format_file_size(size)
    match = SIZE_PATTERN.match(formatted)
    assert match is not None

    _, unit = match.groups()

    if size < 1024:
        assert unit == "bytes", f"Expected 'bytes' for {size}, got {unit!r}"
    elif size < 1024 * 1024:
        assert unit == "KB", f"Expected 'KB' for {size}, got {unit!r}"
    else:
        assert unit == "MB", f"Expected 'MB' for {size}, got {unit!r}"


@settings(max_examples=100)
@given(size=st.integers(min_value=0, max_value=MAX_FILE_SIZE * 2))
def test_format_file_size_non_negative_numeric(size):
    """The numeric part of the formatted string is always non-negative."""
    formatted = format_file_size(size)
    match = SIZE_PATTERN.match(formatted)
    assert match is not None

    numeric_value = float(match.group(1))
    assert numeric_value >= 0, f"Negative numeric value in formatted string: {formatted!r}"
