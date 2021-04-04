# %%
from shutil import rmtree

import pandas as pd

from tensorboard_reducer import read_tb_events, reduce_events, write_csv

reduce_ops = ["mean", "std", "median"]


def test_write_csv():
    rmtree("tmp.csv", ignore_errors=True)

    events_dict = read_tb_events("tests/tensorboard_runs/run_*")

    reduced_events = reduce_events(events_dict, reduce_ops)

    write_csv(reduced_events, "tmp.csv")

    df = pd.read_csv("tmp.csv", header=[0, 1], index_col=0)

    orig_len = len(list(events_dict.values())[0])

    assert len(df) == orig_len, (
        f"length mismatch between original data ({orig_len} timesteps) "
        f"and CSV written to disk ({len(df)} timesteps)"
    )

    rmtree("tmp.csv")
