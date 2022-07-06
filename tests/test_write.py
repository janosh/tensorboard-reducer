from __future__ import annotations

import os
from os.path import isdir
from shutil import rmtree

import pandas as pd
import pytest

import tensorboard_reducer as tbr

from .conftest import REDUCE_OPS


def test_write_tb_events(events_dict: dict[str, pd.DataFrame]) -> None:
    for op in REDUCE_OPS:
        rmtree(f"tmp/reduced-{op}", ignore_errors=True)

    reduced_events = tbr.reduce_events(events_dict, REDUCE_OPS)

    tbr.write_tb_events(reduced_events, "tmp/reduced")

    for op in REDUCE_OPS:
        assert isdir(f"tmp/reduced-{op}"), f"couldn't find {op} reduction outdir"

    tbr.write_tb_events(reduced_events, "tmp/reduced", overwrite=True)

    # will clean up or raise FileNotFoundError if directory unexpectedly does not exist
    for op in REDUCE_OPS:
        rmtree(f"tmp/reduced-{op}")


@pytest.mark.parametrize(
    "ext", [".csv", ".json", ".csv.gz", ".json.gz", ".xls", ".xlsx"]
)
def test_write_data_file(events_dict: dict[str, pd.DataFrame], ext: str) -> None:
    if os.path.exists(f"tmp/strict{ext}"):
        os.remove(f"tmp/strict{ext}")

    reduced_events = tbr.reduce_events(events_dict, REDUCE_OPS)

    tbr.write_data_file(reduced_events, f"tmp/strict{ext}")

    if ".csv" in ext:
        df = pd.read_csv(f"tmp/strict{ext}", header=[0, 1], index_col=0)
    elif ".json" in ext:
        df = pd.read_json(f"tmp/strict{ext}")
    elif ".xls" in ext:
        df = pd.read_excel(f"tmp/strict{ext}", header=[0, 1], index_col=0)

    orig_len = len(list(events_dict.values())[0])  # get step count from logs

    assert len(df) == orig_len, (
        f"length mismatch between original data ({orig_len} timesteps) "
        f"and CSV written to disk ({len(df)} timesteps)"
    )

    # assert no error when overwriting
    tbr.write_data_file(reduced_events, f"tmp/strict{ext}", overwrite=True)

    os.remove(f"tmp/strict{ext}")


def test_write_data_file_with_bad_ext(
    reduced_events: dict[str, dict[str, pd.DataFrame]]
) -> None:
    with pytest.raises(ValueError, match="has unknown extension, should be one of"):
        tbr.write_data_file(reduced_events, "foo.bad_ext")


def test_write_df() -> None:
    with pytest.raises(NotImplementedError, match=r"write_df\(\) was renamed"):
        tbr.write_df(None, "tmp/strict.csv")
