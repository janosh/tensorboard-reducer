# %%
import os

import pandas as pd

from tensorboard_reducer import reduce_events, write_csv


reduce_ops = ["mean", "std", "median"]


def test_write_csv(events_dict):
    if os.path.exists("tmp/strict.csv"):
        os.remove("tmp/strict.csv")

    reduced_events = reduce_events(events_dict, reduce_ops)

    write_csv(reduced_events, "tmp/strict.csv")

    df = pd.read_csv("tmp/strict.csv", header=[0, 1], index_col=0)

    orig_len = len(list(events_dict.values())[0])

    assert len(df) == orig_len, (
        f"length mismatch between original data ({orig_len} timesteps) "
        f"and CSV written to disk ({len(df)} timesteps)"
    )

    write_csv(reduced_events, "tmp/strict.csv", overwrite=True)

    os.remove("tmp/strict.csv")
