from __future__ import annotations

from glob import glob

import pandas as pd
import pytest

from tensorboard_reducer import load_tb_events

lax_runs = glob("tests/runs/lax/run_*")
dup_steps_runs = glob("tests/runs/duplicate_steps/run_*")


@pytest.mark.parametrize("verbose", [True, False])
def test_load_tb_events_strict(
    *, verbose: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test load_tb_events for strict input data, i.e. without any of the special cases
    below.

    The events_dict fixture is the output of load_tb_events() (see conftest.py).
    """
    events_dict = load_tb_events(glob("tests/runs/strict/run_*"), verbose=verbose)
    actual_type = type(events_dict)
    assert_dict = f"return type of load_tb_events() is {actual_type}, expected dict"
    assert actual_type is dict, assert_dict

    actual_keys = list(events_dict)
    assert_keys = (
        f"load_tb_events() returned dict keys {actual_keys}, expected ['strict/foo']"
    )
    assert actual_keys == ["strict/foo"], assert_keys

    n_steps, n_runs = events_dict["strict/foo"].shape
    n_expected = 100
    assert_len = (
        f"load_tb_events() returned TB event with {n_steps} steps, {n_expected=}"
    )
    assert n_steps == n_expected, assert_len

    assert_len = f"load_tb_events() returned {n_runs} TB runs, expected 3"
    assert n_steps == n_expected, assert_len

    # columns correspond to different runs for the same tag, the mean across a run is
    # meaningless and only used for asserting value constancy
    # sorted since order depends on filesystem and is different on Windows
    column_means = sorted(events_dict["strict/foo"].mean(0))

    assert_means = f"load_tb_events() data has unexpected mean {column_means}"
    assert column_means == pytest.approx([1.488, 2.459, 3.481], abs=1e-3), assert_means

    stdout, stderr = capsys.readouterr()
    if verbose:
        assert "Loaded 3 TensorBoard runs with 1 scalars and 100 steps each" in stdout
        assert "\rLoading runs:   0%|" in stderr
        assert "Reading tags:   0%|" in stderr

        # Also test non-strict (lax) settings with verbose=True
        events_dict_lax = load_tb_events(
            lax_runs,
            strict_steps=False,
            strict_tags=False,
            verbose=verbose,
        )
        stdout, stderr = capsys.readouterr()
        assert (
            "Loaded data for 5 tags into arrays of shape (n_steps, n_runs):" in stdout
        )
        for tag in events_dict_lax:
            assert f"- '{tag}'" in stdout
    else:
        assert stdout == ""
        assert stderr == ""


def test_load_tb_events_lax_tags() -> None:
    """Ensure load_tb_events throws an error on runs with different step counts when not
    setting strict_steps=False.
    """
    with pytest.raises(ValueError, match="Unequal number of steps"):
        load_tb_events(lax_runs, strict_tags=False)


def test_load_tb_events_lax_steps() -> None:
    """Ensure load_tb_events throws an error on runs with different sets of tags when
    not setting strict_tags=False.
    """
    with pytest.raises(ValueError, match="Some tags are in some logs but not others"):
        load_tb_events(lax_runs, strict_steps=False)


def test_load_tb_events_lax_tags_and_steps() -> None:
    """Test loading TensorBoard event files when both different sets of tags and
    different step counts across runs should not throw errors.
    """
    events_dict = load_tb_events(lax_runs, strict_tags=False, strict_steps=False)

    tags_list = ["lax/bar_1", "lax/bar_2", "lax/bar_3", "lax/bar_4", "lax/foo"]
    assert sorted(events_dict) == tags_list

    df_lens = [110, 110, 110, 120, 130]
    assert sorted(len(df) for df in events_dict.values()) == df_lens, (
        "Unexpected dataframe lengths"
    )


def test_load_tb_events_handle_dup_steps() -> None:
    """Test loading TensorBoard event files with duplicate steps, i.e. multiple values
    for the same tag at the same step (see handle_dup_steps kwarg).
    """
    with pytest.raises(ValueError, match="contains duplicate steps"):
        load_tb_events(dup_steps_runs)

    kept_first_dups = load_tb_events(dup_steps_runs, handle_dup_steps="keep-first")
    kept_last_dups = load_tb_events(dup_steps_runs, handle_dup_steps="keep-last")
    mean_dups = load_tb_events(dup_steps_runs, handle_dup_steps="mean")

    assert list(kept_first_dups) == list(kept_last_dups) == list(mean_dups), (
        "key mismatch between first, last and mean duplicate handling"
    )

    df_first, df_last, df_mean = (
        dic["dup_steps/foo"] for dic in [kept_first_dups, kept_last_dups, mean_dups]
    )

    # taking the average of keeping first and last duplicates should give same result
    # as taking the mean of duplicate steps
    pd.testing.assert_frame_equal((df_first + df_last) / 2, df_mean)


def test_load_tb_events_min_runs_per_step() -> None:
    """Test loading TensorBoard event files with a minimum number of runs set at which
    to keep steps and below which to drop them (see min_runs_per_step kwarg).
    """
    events_dict = load_tb_events(
        lax_runs,
        strict_steps=False,
        strict_tags=False,
        min_runs_per_step=10,
    )

    # no step has recordings from 10 runs so all dataframes should have 0 length
    assert sum(len(df) for df in events_dict.values()) == 0, (
        "Unexpected non-zero dataframe length for min runs to keep steps=10"
    )

    events_dict = load_tb_events(
        lax_runs,
        strict_steps=False,
        strict_tags=False,
        min_runs_per_step=3,
    )

    # only 1 tag (lax/foo) has recordings from 3 separate runs, lax/bar_1-4 have less
    min_3_lens = [0, 0, 0, 0, 110]
    assert sorted(len(df) for df in events_dict.values()) == min_3_lens, (
        "Unexpected dataframe lengths for min runs to keep steps=3"
    )

    events_dict = load_tb_events(
        lax_runs,
        strict_steps=False,
        strict_tags=False,
        min_runs_per_step=2,
    )

    # 3 tags (lax/foo, lax/bar_2+3) have recordings from at least 2 runs,
    # lax/bar_1+4 have less
    min_2_lens = [0, 0, 110, 120, 120]
    assert (sorted(len(df) for df in events_dict.values())) == min_2_lens, (
        "Unexpected dataframe length for min runs to keep steps=2"
    )


def test_load_tb_events_invalid_inputs() -> None:
    """Test load_tb_events raises for invalid inputs."""
    # ValueError when given empty list of input directories
    with pytest.raises(
        ValueError, match="Expected non-empty list of input directories"
    ):
        load_tb_events([])

    # ValueError when given invalid handle_dup_steps value
    with pytest.raises(ValueError, match="unexpected handle_dup_steps="):
        load_tb_events(glob("tests/runs/strict/run_*"), handle_dup_steps="invalid")

    # ValueError when given invalid min_runs_per_step value
    for r_min in (0, -1):
        with pytest.raises(ValueError, match="Expected positive integer or None"):
            load_tb_events(
                lax_runs, strict_steps=False, strict_tags=False, min_runs_per_step=r_min
            )
