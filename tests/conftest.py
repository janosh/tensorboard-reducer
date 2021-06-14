import os
from glob import glob

import pytest

from tensorboard_reducer import load_tb_events


# https://docs.pytest.org/en/6.2.x/fixture.html#scope-sharing-fixtures-across-classes-modules-packages-or-session
# load events_dict once and reuse across test files for speed
@pytest.fixture(scope="module")
def events_dict():
    os.makedirs("tmp", exist_ok=True)

    tb_events_dict = load_tb_events(glob("tests/runs/strict/run_*"))

    return tb_events_dict
