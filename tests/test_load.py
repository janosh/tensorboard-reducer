from glob import glob

import numpy as np
import pytest

from tensorboard_reducer import load_tb_events

lax_runs = glob("tests/runs/lax/run_*")
dup_steps_runs = glob("tests/runs/duplicate_steps/run_*")


def test_load_tb_events_strict(events_dict):
    """
    Test load_tb_events for strict input data, i.e. without any of the special cases
    below. events_dict is just the output of load_tb_events() (see conftest.py).
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

    with pytest.raises(AssertionError, match="Unequal number of steps"):
        load_tb_events(lax_runs, strict_tags=False)


def test_load_tb_events_lax_steps():
    """Ensure load_tb_events throws an error on runs with different sets of tags when
    not setting strict_tags=False.
    """

    with pytest.raises(AssertionError, match="Some tags appear in some"):
        load_tb_events(lax_runs, strict_steps=False)


def test_load_tb_events_lax_tags_and_steps():
    """
    Test loading TensorBoard event files when both different sets of tags and
    different step counts across runs should not throw errors.
    """

    events_dict = load_tb_events(lax_runs, strict_tags=False, strict_steps=False)

    tags_list = ["lax/bar_1", "lax/bar_2", "lax/bar_3", "lax/bar_4", "lax/foo"]
    assert sorted(events_dict.keys()) == tags_list

    df_lens = [110, 110, 110, 120, 130]
    assert (
        sorted(len(df) for df in events_dict.values()) == df_lens
    ), "Unexpected dataframe lengths"


def test_load_tb_events_handle_dup_steps():
    """
    Test loading TensorBoard event files with duplicate steps, i.e. multiple values
    for the same tag at the same step. See handle_dup_steps kwarg.
    """

    with pytest.raises(AssertionError, match="contains duplicate steps"):
        load_tb_events(dup_steps_runs)

    kept_first_dups = load_tb_events(dup_steps_runs, handle_dup_steps="keep-first")
    kept_last_dups = load_tb_events(dup_steps_runs, handle_dup_steps="keep-last")
    mean_dups = load_tb_events(dup_steps_runs, handle_dup_steps="mean")

    assert (
        kept_first_dups.keys() == kept_last_dups.keys() == mean_dups.keys()
    ), "key mismatch between first, last and mean duplicate handling"

    first_df, last_df, mean_df = (
        dic["dup_steps/foo"] for dic in [kept_first_dups, kept_last_dups, mean_dups]
    )

    assert np.allclose((first_df + last_df) / 2, mean_df, atol=1e-3), (
        "taking the average of keeping first and last duplicates gave different result "
        "than taking the mean of duplicate steps"
    )


def test_load_tb_events_min_runs_per_step():
    """
    Test loading TensorBoard event files with a minimum number of runs set at which
    to keep steps and below which to drop them. See min_runs_per_step kwarg.
    """
    events_dict = load_tb_events(
        lax_runs,
        strict_steps=False,
        strict_tags=False,
        min_runs_per_step=10,
    )

    # no step has recordings from 10 runs so all dataframes should have 0 length
    assert (
        sum(len(df) for df in events_dict.values()) == 0
    ), "Unexpected non-zero dataframe length for min runs to keep steps=10"

    events_dict = load_tb_events(
        lax_runs,
        strict_steps=False,
        strict_tags=False,
        min_runs_per_step=3,
    )

    # only 1 tag (lax/foo) has recordings from 3 separate runs, lax/bar_1-4 have less
    min_3_lens = [0, 0, 0, 0, 110]
    assert (
        sorted(len(df) for df in events_dict.values()) == min_3_lens
    ), "Unexpected dataframe lengths for min runs to keep steps=3"

    events_dict = load_tb_events(
        lax_runs,
        strict_steps=False,
        strict_tags=False,
        min_runs_per_step=2,
    )

    # 3 tags (lax/foo, lax/bar_2+3) have recordings from at least 2 runs,
    # lax/bar_1+4 have less
    min_2_lens = [0, 0, 110, 120, 120]
    assert (
        sorted(len(df) for df in events_dict.values())
    ) == min_2_lens, "Unexpected dataframe length for min runs to keep steps=2"
