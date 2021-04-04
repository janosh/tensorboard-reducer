from tensorboard_reducer import read_tb_events


def test_read_events():
    events_dict = read_tb_events("tests/tensorboard_runs/run_*")

    assert type(events_dict) == dict, "return type of read_tb_events is not dict"
    assert list(events_dict.keys()) == [
        "train/loss"
    ], "read_tb_events returned unexpected dict keys"
    assert (
        len(events_dict["train/loss"]) == 50
    ), "read_tb_events returned unexpected TB event length"
