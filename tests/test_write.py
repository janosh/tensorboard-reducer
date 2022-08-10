from __future__ import annotations

import ast
import itertools
import os
from pathlib import Path

import pandas as pd
import pytest

import tensorboard_reducer as tbr
from tests.conftest import REDUCE_OPS


def test_write_tb_events(
    reduced_events: dict[str, dict[str, pd.DataFrame]], tmp_path: Path
) -> None:
    out_dir = f"{tmp_path}/reduced"
    tbr.write_tb_events(reduced_events, out_dir)

    for op in REDUCE_OPS:
        if op == "std":
            continue
        assert os.path.isdir(f"{out_dir}-{op}"), f"couldn't find {op} reduction out_dir"
    if "std" in REDUCE_OPS:
        assert os.path.isdir(
            f"{out_dir}-mean+std"
        ), "couldn't find mean+std reduction out_dir"
        assert os.path.isdir(
            f"{out_dir}-mean-std"
        ), "couldn't find mean-std reduction out_dir"

    out_dirs = tbr.write_tb_events(reduced_events, out_dir, overwrite=True)

    assert (
        len(out_dirs) == len(REDUCE_OPS) + 1
        if {"mean", "std"}.issubset(REDUCE_OPS)
        else len(REDUCE_OPS)
    )


@pytest.mark.parametrize(
    "extension", [".csv", ".json", ".csv.gz", ".json.gz", ".xls", ".xlsx"]
)
def test_write_data_file(
    reduced_events: dict[str, dict[str, pd.DataFrame]], extension: str, tmp_path: Path
) -> None:
    file_path = f"{tmp_path}/strict{extension}"
    tbr.write_data_file(reduced_events, file_path)

    if ".csv" in extension:
        df = pd.read_csv(file_path, header=[0, 1], index_col=0)
    elif ".json" in extension:
        df = pd.read_json(file_path)
        df.columns = map(ast.literal_eval, df.columns)
    elif ".xls" in extension:
        df = pd.read_excel(file_path, header=[0, 1], index_col=0)

    reduce_ops = list(reduced_events)
    tag_name = list(reduced_events[reduce_ops[0]])
    expected_cols = list(itertools.product(tag_name, reduce_ops))
    n_steps = len(reduced_events[reduce_ops[0]][tag_name[0]])

    assert list(df) == expected_cols, "Unexpected df columns"
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
