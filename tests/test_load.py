import numpy as np
import pytest

from tensorboard_reducer import load_tb_events


def test_load_tb_events_strict(events_dict):
    """Test load_tb_events for strict input data, i.e. without any of the special cases
    below.
    """

    actual_type = type(events_dict)
    assert_dict = f"return type of load_tb_events() is {actual_type}, expected dict"
    assert actual_type == dict, assert_dict

    actual_keys = list(events_dict.keys())
    assert_keys = (
        f"load_tb_events() returned dict keys {actual_keys}, expected ['strict/foo']"
    )
    assert actual_keys == ["strict/foo"], assert_keys

    n_steps, n_runs = events_dict["strict/foo"].shape
    assert_len = (
        f"load_tb_events() returned TB event with {n_steps} steps, expected 100"
    )
    assert n_steps == 100, assert_len

    assert_len = f"load_tb_events() returned {n_runs} TB runs, expected 3"
    assert n_steps == 100, assert_len

    run_means = events_dict["strict/foo"].to_numpy().mean()
    assert_means = f"load_tb_events() data has unexpected mean {run_means:.5}"
    assert np.allclose(run_means, 2.476, atol=1e-3), assert_means


def test_load_tb_events_lax_tags():
    """Ensure load_tb_events throws an error on runs with different step counts when not
    setting strict_steps=False.
    """

    with pytest.raises(Exception) as exc_info:
        load_tb_events("tests/runs/lax/run_*", strict_tags=False)

    assert exc_info.errisinstance(AssertionError), "Unexpected error instance"
    assert (
        "Unequal number of steps" in f"{exc_info}"
    ), "Unexpected error message for load_tb_events(strict_tags=False)"


def test_load_tb_events_lax_steps():
    """Ensure load_tb_events throws an error on runs with different sets of tags when not
    setting strict_tags=False.
    """

    with pytest.raises(Exception) as exc_info:
        load_tb_events("tests/runs/lax/run_*", strict_steps=False)

    assert exc_info.errisinstance(AssertionError), "Unexpected error instance"
    assert (
        "Some tags appear in some" in f"{exc_info}"
    ), "Unexpected error message for load_tb_events(strict_steps=False)"


def test_load_tb_events_lax_tags_and_steps():
    """Test loading TensorBoard event files when both different sets of tags and different
    step counts across runs should not throw errors.
    """

    events_dict = load_tb_events(
        "tests/runs/lax/run_*", strict_tags=False, strict_steps=False
    )

    tags_list = ["lax/bar_1", "lax/bar_2", "lax/bar_3", "lax/bar_4", "lax/foo"]
    assert sorted(events_dict.keys()) == tags_list

    df_lens = [110, 110, 110, 120, 130]
    assert (
        sorted(len(df) for df in events_dict.values()) == df_lens
    ), "Unexpected dataframe lengths"


def test_load_tb_events_handle_dup_steps():
    """Test loading TensorBoard event files with duplicate steps, i.e. multiple values for the
    same tag at the same step.
    """

    with pytest.raises(Exception) as exc_info:
        load_tb_events("tests/runs/duplicate_steps/run_*")

    assert exc_info.errisinstance(AssertionError), "Unexpected error instance"
    assert (
        "contains duplicate steps" in f"{exc_info}"
    ), "Unexpected error message for load_tb_events() when not handling duplicate steps"

    first_dups = load_tb_events(
        "tests/runs/duplicate_steps/run_*", handle_dup_steps="keep-first"
    )
    last_dups = load_tb_events(
        "tests/runs/duplicate_steps/run_*", handle_dup_steps="keep-last"
    )
    mean_dups = load_tb_events(
        "tests/runs/duplicate_steps/run_*", handle_dup_steps="mean"
    )

    assert (
        first_dups.keys() == last_dups.keys() == mean_dups.keys()
    ), "key mismatch between first, last and mean duplicate handling"

    first_df, last_df, mean_df = (
        dic["dup_steps/foo"] for dic in [first_dups, last_dups, mean_dups]
    )

    assert np.allclose((first_df + last_df) / 2, mean_df, atol=1e-3), (
        "taking the average of keeping first and last duplicates gave different result than "
        "taking the mean of duplicate steps"
    )
