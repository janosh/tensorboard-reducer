import os
from os.path import isdir
from shutil import rmtree

import pandas as pd
import pytest

from tensorboard_reducer import reduce_events, write_df, write_tb_events

reduce_ops = ["mean", "std", "median"]


def test_write_tb_events(events_dict):
    for op in reduce_ops:
        rmtree(f"tmp/reduced-{op}", ignore_errors=True)

    reduced_events = reduce_events(events_dict, reduce_ops)

    write_tb_events(reduced_events, "tmp/reduced")

    for op in reduce_ops:
        assert isdir(f"tmp/reduced-{op}"), f"couldn't find {op} reduction outdir"

    write_tb_events(reduced_events, "tmp/reduced", overwrite=True)

    # will clean up or raise FileNotFoundError if directory unexpectedly does not exist
    for op in reduce_ops:
        rmtree(f"tmp/reduced-{op}")


@pytest.mark.parametrize("ext", [".csv", ".json", ".csv.gz", ".json.gz"])
def test_write_df(events_dict, ext):
    if os.path.exists(f"tmp/strict{ext}"):
        os.remove(f"tmp/strict{ext}")

    reduced_events = reduce_events(events_dict, reduce_ops)

    write_df(reduced_events, f"tmp/strict{ext}")

    if ".csv" in ext:
        df = pd.read_csv(f"tmp/strict{ext}", header=[0, 1], index_col=0)
    if ".json" in ext:
        df = pd.read_json(f"tmp/strict{ext}")

    orig_len = len(list(events_dict.values())[0])  # get step count from logs

    assert len(df) == orig_len, (
        f"length mismatch between original data ({orig_len} timesteps) "
        f"and CSV written to disk ({len(df)} timesteps)"
    )

    # assert no error when overwriting
    write_df(reduced_events, f"tmp/strict{ext}", overwrite=True)

    os.remove(f"tmp/strict{ext}")
