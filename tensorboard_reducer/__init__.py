"""TensorBoard Reducer package.

Author: Janosh Riebesell (2021-04-04)
"""

from importlib.metadata import PackageNotFoundError, version

from tensorboard_reducer.load import load_tb_events
from tensorboard_reducer.main import main
from tensorboard_reducer.reduce import reduce_events
from tensorboard_reducer.write import write_data_file, write_tb_events

try:  # noqa: SIM105
    __version__ = version("tensorboard-reducer")
except PackageNotFoundError:
    pass  # package not installed
