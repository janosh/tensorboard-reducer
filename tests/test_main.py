from shutil import rmtree

import pytest

from tensorboard_reducer import main


def test_main():
    """Test main()."""

    rmtree("tmp/strict-mean", ignore_errors=True)

    main(argv=["-i", "tests/runs/strict/run_*", "-o", "tmp/strict"])

    with pytest.raises(Exception) as exc_info:
        main(argv=["-i", "tests/runs/strict/run_*", "-o", "tmp/strict"])

    assert exc_info.errisinstance(FileExistsError), "Unexpected error instance"


def test_main_overwrite():
    main(argv=["-i", "tests/runs/strict/run_*", "-o", "tmp/strict", "-f"])
