"""Quick validation tests for encoding and scaling functions."""
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, r"c:\Users\maury\OneDrive\Desktop\AI Data Analyst\backend")
from cleaning import label_encode, one_hot_encode, standard_scale, min_max_scale


def test_label_encode():
    df = pd.DataFrame({"color": ["red", "blue", "green", "red", "blue"]})
    result = label_encode(df, "color")
    # Expected: blue=0, green=1, red=2 (alphabetical)
    assert list(result["color"]) == [2, 0, 1, 2, 0], f"Got {list(result['color'])}"
    # Original not mutated
    assert list(df["color"]) == ["red", "blue", "green", "red", "blue"]
    print("label_encode PASSED")


def test_label_encode_with_nan():
    df = pd.DataFrame({"color": ["red", None, "blue", "red"]})
    result = label_encode(df, "color")
    # blue=0, red=1 (alphabetical); NaN stays as NaN
    assert result["color"].iloc[0] == 1  # red
    assert pd.isna(result["color"].iloc[1])  # NaN
    assert result["color"].iloc[2] == 0  # blue
    print("label_encode with NaN PASSED")


def test_one_hot_encode():
    df = pd.DataFrame({"color": ["red", "blue", "green"], "value": [1, 2, 3]})
    result = one_hot_encode(df, "color")
    assert "color" not in result.columns
    assert "color_red" in result.columns
    assert "color_blue" in result.columns
    assert "color_green" in result.columns
    # Row 0 was 'red'
    assert result.loc[0, "color_red"] == 1
    assert result.loc[0, "color_blue"] == 0
    assert result.loc[0, "color_green"] == 0
    # Each row has exactly one 1
    dummy_cols = [c for c in result.columns if c.startswith("color_")]
    for i in range(len(result)):
        assert result.iloc[i][dummy_cols].sum() == 1
    print("one_hot_encode PASSED")


def test_one_hot_encode_too_many():
    df = pd.DataFrame({"x": [str(i) for i in range(51)]})
    try:
        one_hot_encode(df, "x")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "50" in str(e)
    print("one_hot_encode >50 categories error PASSED")


def test_standard_scale():
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
    result = standard_scale(df, "x")
    assert abs(result["x"].mean()) < 1e-10
    assert abs(result["x"].std(ddof=0) - 1.0) < 1e-10
    # Original not mutated
    assert df["x"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]
    print("standard_scale PASSED")


def test_standard_scale_zero_std():
    df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
    try:
        standard_scale(df, "x")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "zero standard deviation" in str(e)
    print("standard_scale zero std error PASSED")


def test_standard_scale_non_numerical():
    df = pd.DataFrame({"x": ["a", "b", "c"]})
    try:
        standard_scale(df, "x")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not numerical" in str(e)
    print("standard_scale non-numerical error PASSED")


def test_min_max_scale():
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
    result = min_max_scale(df, "x")
    assert result["x"].min() == 0.0
    assert result["x"].max() == 1.0
    # Check intermediate value
    assert abs(result["x"].iloc[2] - 0.5) < 1e-10  # 30 is midpoint
    # Original not mutated
    assert df["x"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]
    print("min_max_scale PASSED")


def test_min_max_scale_constant():
    df = pd.DataFrame({"x": [7.0, 7.0, 7.0]})
    try:
        min_max_scale(df, "x")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "identical min and max" in str(e)
    print("min_max_scale min==max error PASSED")


def test_min_max_scale_non_numerical():
    df = pd.DataFrame({"x": ["a", "b"]})
    try:
        min_max_scale(df, "x")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not numerical" in str(e)
    print("min_max_scale non-numerical error PASSED")


def test_nonexistent_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    for fn in [label_encode, one_hot_encode, standard_scale, min_max_scale]:
        try:
            fn(df, "nonexistent")
            assert False, f"{fn.__name__} should have raised ValueError"
        except ValueError:
            pass
    print("nonexistent column errors PASSED")


if __name__ == "__main__":
    test_label_encode()
    test_label_encode_with_nan()
    test_one_hot_encode()
    test_one_hot_encode_too_many()
    test_standard_scale()
    test_standard_scale_zero_std()
    test_standard_scale_non_numerical()
    test_min_max_scale()
    test_min_max_scale_constant()
    test_min_max_scale_non_numerical()
    test_nonexistent_column()
    print("\n=== ALL TESTS PASSED ===")
