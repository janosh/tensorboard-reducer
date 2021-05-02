![TensorBoard Reducer](https://raw.githubusercontent.com/janosh/tensorboard-reducer/main/assets/tensorboard-reducer.svg)

[![Tests](https://github.com/janosh/tensorboard-reducer/workflows/Tests/badge.svg)](https://github.com/janosh/tensorboard-reducer/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/tensorboard-reducer/main.svg)](https://results.pre-commit.ci/latest/github/janosh/tensorboard-reducer/main)
[![PyPI](https://img.shields.io/pypi/v/tensorboard-reducer)](https://pypi.org/project/tensorboard-reducer)
[![This project supports Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/downloads)

> This project was inspired by [`tensorboard-aggregator`](https://github.com/Spenhouet/tensorboard-aggregator) (similar project built for TensorFlow rather than PyTorch) and [this SO answer](https://stackoverflow.com/a/48774926).

Compute reduced statistics (`mean`, `std`, `min`, `max`, `median` or any other [`numpy`](https://numpy.org/doc/stable) operation) of multiple TensorBoard runs matching a directory glob pattern. This can e.g. be used when training multiple identical models to reduce the noise in their loss/accuracy/error curves to establish statistical significance in performance improvements. The aggregation results can be saved to disk either as new TensorBoard event files or as a CSV.

Requires [PyTorch](https://pypi.org/project/torch) and [TensorBoard](https://pypi.org/project/tensorboard). No TensorFlow installation required.

## Installation

```sh
pip install tensorboard-reducer
```

## Usage

### Through CLI

```sh
tb-reducer -i 'glob_pattern/of_dirs_to_reduce*' -o basename_of_output_dir -r mean,std,min,max
```

![Mean of 3 TensorBoard logs](https://raw.githubusercontent.com/janosh/tensorboard-reducer/main/assets/3-runs-mean.png)

`tb-reducer` has the following flags:

- `-i/--indirs-glob` (required): Glob pattern of the run directories to reduce.
- `-o/--outdir` (required): Name of the directory to save the new reduced run data. If `--format` is `tb-events`, a separate directory will be created for each reduce op (`mean`, `std`, ...) suffixed by the op's name (`outdir-mean`, `outdir-std`, ...). If `--format` is `csv`, a single file will created and `outdir` must end with a `.csv` extension.
- `-r/--reduce-ops` (optional): Comma-separated names of numpy reduction ops (`mean`, `std`, `min`, `max`, ...). Default is `mean`. Each reduction is written to a separate `outdir` suffixed by its op name, e.g. if `outdir='my-new-run`, the mean reduction will be written to `my-new-run-mean`.
- `-f/--format`: Output format of reduced TensorBoard runs. One of `tb-events` for regular TensorBoard event files or `csv`. If `csv`, `-o/--outdir` must have `.csv` extension and all reduction ops will be written to a single CSV file rather than separate directories for each reduce op. Use `pandas.read_csv("path/to/file.csv", header=[0, 1], index_col=0)` to read data back into memory as a multi-index dataframe.
- `-w/--overwrite` (optional): Whether to overwrite existing `outdir`s/CSV files.

### Through Python API

You can also import `tensorboard_reducer` into a Python script for more complex operations. A simple example to get you started:

```py
from tensorboard_reducer import load_tb_events, write_tb_events, reduce_events

indirs_glob = "glob_pattern/of_directories_to_reduce*"
outdir = "path/to/output_dir"
overwrite = False
reduce_ops = ["mean", "min", "max"]

events_dict = load_tb_events(indirs_glob)

n_steps, n_events = list(events_dict.values())[0].shape
n_scalars = len(events_dict)

print(
    f"Loaded {n_events} TensorBoard runs with {n_scalars} scalars and {n_steps} steps each"
)
for tag in events_dict.keys():
    print(f" - {tag}")

reduced_events = reduce_events(events_dict, reduce_ops)

for op in reduce_ops:
    print(f"Writing '{op}' reduction to '{outdir}-{op}'")

write_tb_events(reduced_events, outdir, overwrite)
```

## Doc Strings

```py
def reduce_events(
    events_dict: Dict[str, Array], reduce_ops: List[str]
) -> Dict[str, Dict[str, Array]]:
    """Perform numpy reduce ops on the last axis of each array
    in a dictionary of scalar TensorBoard event data. Each array enters
    this function with shape (n_timesteps, r_runs) and len(reduce_ops) exit
    with shape (n_timesteps,).

    Args:
        events_dict (dict[str, Array]): Dictionary of arrays to reduce.
        reduce_ops (list[str]): numpy reduce ops.

    Returns:
        dict[str, dict[str, Array]]: Dict of dicts where each subdict holds one reduced array
            for each of the specified reduce ops, e.g. {"loss": {"mean": arr.mean(-1),
            "std": arr.std(-1)}}.
    """
```

```py
def load_tb_events(indirs_glob: str) -> Dict[str, Array]:
    """Read the TensorBoard event files matching the provided glob pattern
    and return their scalar data as a dict with tags ('training/loss',
    'validation/mae', etc.) as keys and 2d arrays of shape (n_timesteps, r_runs)
    as values.

    Args:
        indirs_glob (str): Glob pattern of the run directories to read from disk.

    Returns:
        dict: A dictionary of containing scalar run data with keys like
            'train/loss', 'train/mae', 'val/loss', etc.
    """
```

```py
def write_tb_events(
    data_to_write: Dict[str, Dict[str, Array]],
    outdir: str,
    overwrite: bool = False,
) -> None:
    """Writes data in dict to disk as TensorBoard event files in a newly created/overwritten
    outdir directory.

    Inspired by https://stackoverflow.com/a/48774926.

    Args:
        data_to_write (dict[str, dict[str, Array]]): Data to write to disk. Assumes 1st-level
            keys are reduce ops (mean, std, ...) and 2nd-level are TensorBoard tags.
        outdir (str): Name of the directory to save the new reduced run data. Will
            have the reduce op name (e.g. '-mean'/'-std') appended.
        overwrite (bool): Whether to overwrite existing reduction directories.
            Defaults to False.
    """
```

```py
def write_csv(
    data_to_write: Dict[str, Dict[str, Array]],
    csv_path: str,
    overwrite: bool = False,
) -> None:
    """Writes reduced TensorBoard data passed as dict of dicts (1st arg) to a CSV file
    path (2nd arg).

    Use pd.read_csv("path/to/file.csv", header=[0, 1], index_col=0) to read data back into
    memory as a multi-index dataframe.

    Args:
        data_to_write (dict[str, dict[str, Array]]): Data to write to disk. Assumes 1st-level
            keys are reduce ops (mean, std, ...) and 2nd-level are TensorBoard tags.
        outdir (str): Name of the directory to save the new reduced run data. Will
            have the reduce op name (e.g. '-mean'/'-std') appended.
        overwrite (bool): Whether to overwrite existing reduction directories.
            Defaults to False.
    """
```
