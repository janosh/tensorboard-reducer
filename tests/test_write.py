from __future__ import annotations

import ast
import itertools
import os
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import tensorboard_reducer as tbr
from tests.conftest import REDUCE_OPS

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("verbose", [True, False])
def test_write_tb_events(
    reduced_events: dict[str, dict[str, pd.DataFrame]],
    tmp_path: Path,
    verbose: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = str(tmp_path)
    tbr.write_tb_events(reduced_events, out_dir, verbose=verbose)

    for op in REDUCE_OPS:
        if op == "std":
            continue
        assert os.path.isdir(f"{out_dir}-{op}"), f"couldn't find {op = } out_dir"
    if "std" in REDUCE_OPS:
        assert os.path.isdir(f"{out_dir}-mean+std"), (
            "couldn't find mean+std reduction out_dir"
        )
        assert os.path.isdir(f"{out_dir}-mean-std"), (
            "couldn't find mean-std reduction out_dir"
        )

    out_dirs = tbr.write_tb_events(reduced_events, out_dir, overwrite=True)

    assert (
        len(out_dirs) == len(REDUCE_OPS) + 1
        if {"mean", "std"}.issubset(REDUCE_OPS)
        else len(REDUCE_OPS)
    )
    stdout, stderr = capsys.readouterr()
    if verbose:
        for op in REDUCE_OPS:
            if op == "std" and "mean" in REDUCE_OPS:
                assert "Writing mean+std reduction to disk..." in stderr
                assert "Writing mean-std reduction to disk..." in stderr
            else:
                assert f"Writing {op} reduction to disk:" in stderr
        assert "Created new TensorBoard event files in\n" in stdout
        for out_dir in out_dirs:
            assert f"\n- {out_dir}" in stdout
    else:
        assert stdout == ""
        assert stderr == ""


@pytest.mark.parametrize("extension", [".csv", ".json", ".csv.gz", ".json.gz", ".xlsx"])
@pytest.mark.parametrize("verbose", [True, False])
def test_write_data_file(
    reduced_events: dict[str, dict[str, pd.DataFrame]],
    extension: str,
    tmp_path: Path,
    verbose: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test writing reduced events to different file formats."""
    if extension == ".xlsx":
        pytest.importorskip("openpyxl")

    file_path = f"{tmp_path}/strict{extension}"
    tbr.write_data_file(reduced_events, file_path, verbose=verbose)

    if ".csv" in extension:
        df_actual = pd.read_csv(file_path, header=[0, 1], index_col=0)
    elif ".json" in extension:
        df_actual = pd.read_json(file_path)
        df_actual.columns = map(ast.literal_eval, df_actual.columns)
    elif ".xlsx" in extension:
        df_actual = pd.read_excel(file_path, header=[0, 1], index_col=0)
    else:
        raise ValueError(f"Unknown {extension=}")

    reduce_ops = list(reduced_events)
    tag_name = list(reduced_events[reduce_ops[0]])
    expected_cols = list(itertools.product(tag_name, reduce_ops))
    n_steps = len(reduced_events[reduce_ops[0]][tag_name[0]])

    assert list(df_actual) == expected_cols, "Unexpected df columns"
    assert df_actual.shape == (n_steps, len(reduce_ops)), "Unexpected df shape"

    out_path = tbr.write_data_file(reduced_events, file_path, overwrite=True)

    stdout, stderr = capsys.readouterr()
    assert stderr == ""
    if verbose:
        assert f"Created new data file at {out_path!r}" in stdout
    else:
        assert stdout == ""


def test_write_data_file_with_bad_ext(
    reduced_events: dict[str, dict[str, pd.DataFrame]],
) -> None:
    with pytest.raises(ValueError, match="has unknown extension, should be one of"):
        tbr.write_data_file(reduced_events, "foo.bad_ext")
