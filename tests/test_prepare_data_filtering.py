import pytest

from analysis.prepare_data import training_indices


def test_training_indices_without_exclusions_preserve_the_existing_split():
    assert training_indices(8, [1, 6]) == [0, 2, 3, 4, 5, 7]


def test_training_indices_exclude_original_rows_after_the_diagnostic_split():
    assert training_indices(8, [1, 6], [3, 7]) == [0, 2, 4, 5]


def test_training_indices_reject_excluding_a_diagnostic_row():
    with pytest.raises(ValueError, match="cannot exclude diagnostic rows"):
        training_indices(8, [1, 6], [6])


def test_training_indices_reject_out_of_range_source_indices():
    with pytest.raises(ValueError, match="out of range"):
        training_indices(8, [1, 6], [8])
