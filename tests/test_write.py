from __future__ import annotations

import ast
import itertools
from os.path import isdir
from pathlib import Path

import pandas as pd
import pytest

import tensorboard_reducer as tbr

from .conftest import REDUCE_OPS


def test_write_tb_events(
    reduced_events: dict[str, dict[str, pd.DataFrame]], tmp_path: Path
) -> None:
    outdir = f"{tmp_path}/reduced"
    tbr.write_tb_events(reduced_events, outdir)

    for op in REDUCE_OPS:
        assert isdir(f"{outdir}-{op}"), f"couldn't find {op} reduction outdir"

    tbr.write_tb_events(reduced_events, outdir, overwrite=True)

    for op in REDUCE_OPS:
        assert isdir(f"{outdir}-{op}"), f"couldn't find {op} reduction outdir"


@pytest.mark.parametrize(
    "ext", [".csv", ".json", ".csv.gz", ".json.gz", ".xls", ".xlsx"]
)
def test_write_data_file(
    reduced_events: dict[str, dict[str, pd.DataFrame]], ext: str, tmp_path: Path
) -> None:
    file_path = f"{tmp_path}/strict{ext}"
    tbr.write_data_file(reduced_events, file_path)

    if ".csv" in ext:
        df = pd.read_csv(file_path, header=[0, 1], index_col=0)
    elif ".json" in ext:
        df = pd.read_json(file_path)
        df.columns = map(ast.literal_eval, df.columns)
    elif ".xls" in ext:
        df = pd.read_excel(file_path, header=[0, 1], index_col=0)

    reduce_ops = list(reduced_events)
    tag_name = list(reduced_events[reduce_ops[0]])
    expected_cols = list(itertools.product(tag_name, reduce_ops))
    n_steps = len(reduced_events[reduce_ops[0]][tag_name[0]])

    assert list(df.columns) == expected_cols, "Unexpected df columns"
    assert df.shape == (n_steps, len(reduce_ops)), "Unexpected df shape"

    tbr.write_data_file(reduced_events, file_path, overwrite=True)


def test_write_data_file_with_bad_ext(
    reduced_events: dict[str, dict[str, pd.DataFrame]]
) -> None:
    with pytest.raises(ValueError, match="has unknown extension, should be one of"):
        tbr.write_data_file(reduced_events, "foo.bad_ext")


def test_write_df() -> None:
    with pytest.raises(NotImplementedError, match=r"write_df\(\) was renamed"):
        tbr.write_df(None, "foo.csv")
