from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import pytest
from pytest import CaptureFixture

from tensorboard_reducer import main

strict_runs = glob("tests/runs/strict/run_*")

lax_runs = glob("tests/runs/lax/run_*")


@pytest.mark.parametrize("outpath_flag", ["--outpath", "-o"])
def test_main(tmp_path: Path, outpath_flag: str) -> None:
    """Test main()."""

    main([*strict_runs, outpath_flag, f"{tmp_path}/strict"])

    # make sure reduction fails if output dir already exists
    with pytest.raises(FileExistsError):
        main([*strict_runs, outpath_flag, f"{tmp_path}/strict"])


@pytest.mark.parametrize("overwrite_flag", ["--overwrite", "-f"])
def test_main_overwrite(tmp_path: Path, overwrite_flag: str) -> None:
    main([*strict_runs, "-o", f"{tmp_path}/strict", overwrite_flag])


@pytest.mark.parametrize("reduce_ops_flag", ["--reduce-ops", "-r"])
def test_main_multi_reduce(tmp_path: Path, reduce_ops_flag: str) -> None:
    reduce_ops = sorted(["mean", "std", "min", "max"])
    outdir = f"{tmp_path}{os.path.sep}strict"

    main([*strict_runs, "-o", outdir, "-f", reduce_ops_flag, ",".join(reduce_ops)])

    # make sure all outdirs were created
    for reduce_op in reduce_ops:
        assert os.path.isdir(f"{outdir}-{reduce_op}")


def test_main_lax(tmp_path: Path) -> None:
    outdir = f"{tmp_path}{os.path.sep}lax"
    with pytest.raises(AssertionError):
        main([*lax_runs, "-o", outdir])

    with pytest.raises(AssertionError):
        main([*lax_runs, "-o", outdir, "--lax-tags"])

    with pytest.raises(AssertionError):
        main([*lax_runs, "-o", outdir, "--lax-steps"])

    main([*lax_runs, "-o", outdir, "--lax-tags", "--lax-steps"])

    # make sure outdir was created
    assert os.path.isdir(f"{outdir}-mean")


def test_main_lax_csv_output(tmp_path: Path) -> None:
    out_file = f"{tmp_path}/lax.csv"

    main([*lax_runs, "-o", out_file, "--lax-tags", "--lax-steps", "-r", "mean,std"])

    # make sure CSV file was created
    assert os.path.isfile(out_file)


@pytest.mark.parametrize("arg", ["-v", "--version"])
def test_main_report_version(capsys: CaptureFixture[str], arg: str) -> None:
    """Test CLI version flag."""

    with pytest.raises(SystemExit):
        ret_code = main([arg])
        assert ret_code == 0

    stdout, stderr = capsys.readouterr()

    assert stdout.startswith("TensorBoard Reducer v")
    assert stderr == ""
