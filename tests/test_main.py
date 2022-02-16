import os
from glob import glob
from shutil import rmtree

import pytest

from tensorboard_reducer import main

strict_runs = glob("tests/runs/strict/run_*")
argv_strict = [*strict_runs, "-o", "tmp/strict"]

lax_runs = glob("tests/runs/lax/run_*")
argv_lax = [*lax_runs, "-o", "tmp/lax"]


def test_main():
    """Test main()."""

    rmtree("tmp/strict-mean", ignore_errors=True)

    main(argv_strict)

    with pytest.raises(FileExistsError):
        main(argv_strict)


def test_main_overwrite():
    main(argv_strict + ["-f"])


def test_main_multi_reduce():
    reduce_ops = ["mean", "std", "min", "max"]

    main(argv_strict + ["-f", "-r", ",".join(reduce_ops)])

    # make sure all outdirs were created
    for op in reduce_ops:
        rmtree(f"tmp/strict-{op}")


def test_main_lax():
    # make sure we start from clean slate in case prev test failed
    rmtree("tmp/lax-mean", ignore_errors=True)

    with pytest.raises(AssertionError):
        main(argv_lax)

    with pytest.raises(AssertionError):
        main(argv_lax + ["--lax-tags"])

    with pytest.raises(AssertionError):
        main(argv_lax + ["--lax-steps"])

    main(argv_lax + ["--lax-tags", "--lax-steps"])

    # make sure outdir was created
    rmtree("tmp/lax-mean")


def test_main_lax_csv_output():
    # make sure we start from clean slate in case prev test failed
    try:
        os.remove("tmp/lax.csv")
    except FileNotFoundError:
        pass

    main(
        [*lax_runs, "-o", "tmp/lax.csv", "--lax-tags", "--lax-steps", "-r", "mean,std"]
    )

    # make sure CSV file was created
    os.remove("tmp/lax.csv")


@pytest.mark.parametrize("arg", ["-v", "--version"])
def test_main_report_version(capsys, arg):
    """Test CLI version flag."""

    with pytest.raises(SystemExit):
        ret_code = main([arg])
        assert ret_code == 0

    stdout, stderr = capsys.readouterr()

    assert stdout.startswith("TensorBoard Reducer v")
    assert stderr == ""
