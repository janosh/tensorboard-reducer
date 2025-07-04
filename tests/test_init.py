from __future__ import annotations

import re

import tensorboard_reducer as tbr


def test_init_imports() -> None:
    """Test that all expected imports are available from tensorboard_reducer."""
    assert callable(tbr.load_tb_events)
    assert callable(tbr.reduce_events)
    assert callable(tbr.write_tb_events)
    assert callable(tbr.write_data_file)
    assert callable(tbr.main)


def test_init_version() -> None:
    """Test that the version is available when the package is installed."""
    assert re.match(r"\d+\.\d+\.\d+", tbr.__version__) is not None
