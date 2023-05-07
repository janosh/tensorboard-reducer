from __future__ import annotations

from glob import glob

import pandas as pd
import pytest

import tensorboard_reducer as tbr

REDUCE_OPS = ("mean", "std", "median")


# load events_dict once and reuse across tests
# https://docs.pytest.org/en/6.2.x/fixture.html#fixture-scopes
@pytest.fixture(scope="session")
def events_dict() -> dict[str, pd.DataFrame]:
    return tbr.load_tb_events(glob("tests/runs/strict/run_*"))


@pytest.fixture(scope="session")
def reduced_events(
    events_dict: dict[str, pd.DataFrame]
) -> dict[str, dict[str, pd.DataFrame]]:
    return tbr.reduce_events(events_dict, REDUCE_OPS)
