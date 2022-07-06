from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import py
import pytest
from pytest import CaptureFixture

from tensorboard_reducer import main

strict_runs = glob("tests/runs/strict/run_*")

lax_runs = glob("tests/runs/lax/run_*")


@pytest.mark.parametrize("outpath_flag", ["--outpath", "-o"])
def test_main(tmpdir: py.path.local, outpath_flag: str) -> None:
    """Test main()."""

    main([*strict_runs, outpath_flag, f"{tmpdir}/strict"])

    # make sure reduction fails if output dir already exists
    with pytest.raises(FileExistsError):
        main([*strict_runs, outpath_flag, f"{tmpdir}/strict"])


@pytest.mark.parametrize("overwrite_flag", ["--overwrite", "-f"])
def test_main_overwrite(tmpdir: py.path.local, overwrite_flag: str) -> None:
    main([*strict_runs, "-o", f"{tmpdir}/strict", overwrite_flag])


@pytest.mark.parametrize("reduce_ops_flag", ["--reduce-ops", "-r"])
def test_main_multi_reduce(tmpdir: py.path.local, reduce_ops_flag: str) -> None:
    reduce_ops = sorted(["mean", "std", "min", "max"])
    outdir = f"{tmpdir}{os.path.sep}strict"

    main([*strict_runs, "-o", outdir, "-f", reduce_ops_flag, ",".join(reduce_ops)])

    # make sure all outdirs were created
    assert [f"{outdir}-{op}" for op in reduce_ops] == sorted(map(str, tmpdir.listdir()))


def test_main_lax(tmpdir: py.path.local) -> None:
    outdir = f"{tmpdir}{os.path.sep}lax"
    with pytest.raises(AssertionError):
        main([*lax_runs, "-o", outdir])

    with pytest.raises(AssertionError):
        main([*lax_runs, "-o", outdir, "--lax-tags"])

    with pytest.raises(AssertionError):
        main([*lax_runs, "-o", outdir, "--lax-steps"])

    main([*lax_runs, "-o", outdir, "--lax-tags", "--lax-steps"])

    # make sure outdir was created
    assert [f"{outdir}-mean"] == tmpdir.listdir()


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
