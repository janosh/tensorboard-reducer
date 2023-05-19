from __future__ import annotations

import os
from glob import glob
from typing import TYPE_CHECKING

import pytest
from pytest import CaptureFixture

from tensorboard_reducer import main

if TYPE_CHECKING:
    from pathlib import Path


strict_runs = glob("tests/runs/strict/run_*")

lax_runs = glob("tests/runs/lax/run_*")


@pytest.mark.parametrize("out_path_flag", ["--outpath", "-o"])
def test_main(tmp_path: Path, out_path_flag: str) -> None:
    """Test main()."""
    main([*strict_runs, out_path_flag, f"{tmp_path}/strict"])

    # make sure reduction fails if output dir already exists
    with pytest.raises(FileExistsError):
        main([*strict_runs, out_path_flag, f"{tmp_path}/strict"])


@pytest.mark.parametrize("overwrite_flag", ["--overwrite", "-f"])
def test_main_overwrite(tmp_path: Path, overwrite_flag: str) -> None:
    main([*strict_runs, "-o", f"{tmp_path}/strict", overwrite_flag])


@pytest.mark.parametrize("reduce_ops_flag", ["--reduce-ops", "-r"])
def test_main_multi_reduce(tmp_path: Path, reduce_ops_flag: str) -> None:
    reduce_ops = sorted(["mean", "std", "min", "max"])
    out_dir = f"{tmp_path}{os.path.sep}strict"

    main([*strict_runs, "-o", out_dir, "-f", reduce_ops_flag, ",".join(reduce_ops)])

    # make sure all out_dirs were created
    for op in reduce_ops:
        if op == "std":
            continue
        assert os.path.isdir(f"{out_dir}-{op}"), f"couldn't find {op} reduction out_dir"
    if "std" in reduce_ops:
        assert os.path.isdir(
            f"{out_dir}-mean+std"
        ), "couldn't find mean+std reduction out_dir"
        assert os.path.isdir(
            f"{out_dir}-mean-std"
        ), "couldn't find mean-std reduction out_dir"


def test_main_lax(tmp_path: Path) -> None:
    out_dir = f"{tmp_path}{os.path.sep}lax"
    with pytest.raises(ValueError, match="Some tags are in some logs but not others"):
        main([*lax_runs, "-o", out_dir])

    with pytest.raises(ValueError, match="Unequal number of steps "):
        main([*lax_runs, "-o", out_dir, "--lax-tags"])

    with pytest.raises(ValueError, match="Some tags are in some logs but not others"):
        main([*lax_runs, "-o", out_dir, "--lax-steps"])

    main([*lax_runs, "-o", out_dir, "--lax-tags", "--lax-steps"])

    # make sure out_dir was created
    assert os.path.isdir(f"{out_dir}-mean")


def test_main_lax_csv_output(tmp_path: Path) -> None:
    out_file = f"{tmp_path}/lax.csv"

    main([*lax_runs, "-o", out_file, "--lax-tags", "--lax-steps", "-r", "mean,std"])

    # make sure CSV file was created
    assert os.path.isfile(out_file)


@pytest.mark.parametrize("arg", ["-v", "--version"])
def test_main_report_version(capsys: CaptureFixture[str], arg: str) -> None:
    """Test CLI version flag."""
    with pytest.raises(SystemExit) as exc_info:
        main([arg])

    assert exc_info.value.code == 0

    stdout, stderr = capsys.readouterr()

    assert stdout.startswith("TensorBoard Reducer v")
    assert stderr == ""
