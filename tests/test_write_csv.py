import os

import pandas as pd
import pytest

from tensorboard_reducer import reduce_events, write_df

reduce_ops = ["mean", "std", "median"]


@pytest.mark.parametrize("ext", [".csv", ".json", ".csv.gz", ".json.gz"])
def test_write_df(events_dict, ext):
    if os.path.exists(f"tmp/strict{ext}"):
        os.remove(f"tmp/strict{ext}")

    reduced_events = reduce_events(events_dict, reduce_ops)

    write_df(reduced_events, f"tmp/strict{ext}")

    df = pd.read_csv(f"tmp/strict{ext}", header=[0, 1], index_col=0)

    orig_len = len(list(events_dict.values())[0])  # get step count from logs

    assert len(df) == orig_len, (
        f"length mismatch between original data ({orig_len} timesteps) "
        f"and CSV written to disk ({len(df)} timesteps)"
    )

    # assert no error when overwriting
    write_df(reduced_events, f"tmp/strict{ext}", overwrite=True)

    os.remove(f"tmp/strict{ext}")
