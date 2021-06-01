import numpy as np
import pytest

from tensorboard_reducer import load_tb_events


def test_read_events_strict(events_dict):

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


def test_read_events_lax_tags():

    with pytest.raises(Exception) as exc_info:
        load_tb_events("tests/runs/lax/run_*", strict_tags=False)

    assert exc_info.errisinstance(AssertionError), "Unexpected error instance"
    assert (
        "Unequal number of steps" in f"{exc_info}"
    ), "Unexpected error message for load_tb_events(strict_tags=False)"


def test_read_events_lax_steps():

    with pytest.raises(Exception) as exc_info:
        load_tb_events("tests/runs/lax/run_*", strict_steps=False)

    assert exc_info.errisinstance(AssertionError), "Unexpected error instance"
    assert (
        "Some tags appear in some" in f"{exc_info}"
    ), "Unexpected error message for load_tb_events(strict_steps=False)"


def test_read_events_lax_tags_and_steps():

    events_dict = load_tb_events(
        "tests/runs/lax/run_*", strict_tags=False, strict_steps=False
    )

    tags_list = ["lax/foo", "lax/bar_2", "lax/bar_3", "lax/bar_4", "lax/bar_1"]
    assert list(events_dict.keys()) == tags_list

    df_lens = [110, 110, 120, 130, 110]
    assert [
        len(df) for df in events_dict.values()
    ] == df_lens, "Unexpected dataframe lengths"
