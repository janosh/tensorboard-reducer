import numpy as np


def test_read_events(events_dict):

    actual_type = type(events_dict)
    assert_dict = f"return type of load_tb_events() is {actual_type}, expected dict"
    assert actual_type == dict, assert_dict

    actual_keys = list(events_dict.keys())
    assert_keys = (
        f"load_tb_events() returned dict keys {actual_keys}, expected ['train/loss']"
    )
    assert actual_keys == ["train/loss"], assert_keys

    n_steps, n_runs = events_dict["train/loss"].shape
    assert_len = f"load_tb_events() returned TB event with {n_steps} steps, expected 50"
    assert n_steps == 50, assert_len

    assert_len = f"load_tb_events() returned {n_runs} TB runs, expected 3"
    assert n_steps == 50, assert_len

    run_means = events_dict["train/loss"].mean()
    assert_means = "load_tb_events() data has unexpected mean"
    assert np.allclose(run_means, 31.28, atol=1e-3), assert_means
