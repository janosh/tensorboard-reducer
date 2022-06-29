from __future__ import annotations

import os
from os.path import isdir
from shutil import rmtree

import pandas as pd
import pytest

import tensorboard_reducer as tbr

reduce_ops = ["mean", "std", "median"]


def test_write_tb_events(events_dict: dict[str, pd.DataFrame]) -> None:
    for op in reduce_ops:
        rmtree(f"tmp/reduced-{op}", ignore_errors=True)

    reduced_events = tbr.reduce_events(events_dict, reduce_ops)

    tbr.write_tb_events(reduced_events, "tmp/reduced")

    for op in reduce_ops:
        assert isdir(f"tmp/reduced-{op}"), f"couldn't find {op} reduction outdir"

    tbr.write_tb_events(reduced_events, "tmp/reduced", overwrite=True)

    # will clean up or raise FileNotFoundError if directory unexpectedly does not exist
    for op in reduce_ops:
        rmtree(f"tmp/reduced-{op}")


@pytest.mark.parametrize("ext", [".csv", ".json", ".csv.gz", ".json.gz"])
def test_write_data_file(events_dict: dict[str, pd.DataFrame], ext: str) -> None:
    if os.path.exists(f"tmp/strict{ext}"):
        os.remove(f"tmp/strict{ext}")

    reduced_events = tbr.reduce_events(events_dict, reduce_ops)

    tbr.write_data_file(reduced_events, f"tmp/strict{ext}")

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
    tbr.write_data_file(reduced_events, f"tmp/strict{ext}", overwrite=True)

    os.remove(f"tmp/strict{ext}")


def test_write_df() -> None:
    with pytest.raises(NotImplementedError, match=r"write_df\(\) was renamed"):
        tbr.write_df(None, "tmp/strict.csv")
