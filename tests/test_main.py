from shutil import rmtree

import pytest

from tensorboard_reducer import main


argv_strict = ["-i", "tests/runs/strict/run_*", "-o", "tmp/strict"]
argv_lax = ["-i", "tests/runs/lax/run_*", "-o", "tmp/lax"]


def test_main():
    """Test main()."""

    rmtree("tmp/strict-mean", ignore_errors=True)

    main(argv_strict)

    with pytest.raises(Exception) as exc_info:
        main(argv_strict)

    assert exc_info.errisinstance(FileExistsError), "Unexpected error instance"


def test_main_overwrite():
    main(argv_strict + ["-f"])


def test_main_multi_reduce():
    main(argv_strict + ["-f", "-r", "mean,std,min,max"])


def test_main_lax():
    # make sure we start from clean slate in case prev test failed
    rmtree("tmp/lax-mean", ignore_errors=True)

    with pytest.raises(Exception) as exc_info:
        main(argv_lax)

    assert exc_info.errisinstance(AssertionError), "Unexpected error instance"

    with pytest.raises(Exception) as exc_info:
        main(argv_lax + ["--lax-tags"])

    assert exc_info.errisinstance(AssertionError), "Unexpected error instance"

    with pytest.raises(Exception) as exc_info:
        main(argv_lax + ["--lax-steps"])

    assert exc_info.errisinstance(AssertionError), "Unexpected error instance"

    main(argv_lax + ["--lax-tags", "--lax-steps"])

    # make sure outdir was created
    rmtree("tmp/strict-mean")
