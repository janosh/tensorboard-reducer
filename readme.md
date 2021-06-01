![TensorBoard Reducer](https://raw.githubusercontent.com/janosh/tensorboard-reducer/main/assets/tensorboard-reducer.svg)

[![Tests](https://github.com/janosh/tensorboard-reducer/workflows/Tests/badge.svg)](https://github.com/janosh/tensorboard-reducer/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/tensorboard-reducer/main.svg)](https://results.pre-commit.ci/latest/github/janosh/tensorboard-reducer/main)
[![PyPI](https://img.shields.io/pypi/v/tensorboard-reducer)](https://pypi.org/project/tensorboard-reducer)
[![This project supports Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/downloads)

> This project was inspired by [`tensorboard-aggregator`](https://github.com/Spenhouet/tensorboard-aggregator) (similar project built for TensorFlow rather than PyTorch) and [this SO answer](https://stackoverflow.com/a/48774926).

Compute reduced statistics (`mean`, `std`, `min`, `max`, `median` or any other [`numpy`](https://numpy.org/doc/stable) operation) of multiple TensorBoard runs matching a directory glob pattern. This can e.g. be used when training multiple identical models to reduce the noise in their loss/accuracy/error curves to establish statistical significance in performance improvements. The aggregation results can be saved to disk either as new TensorBoard event files or as CSV.

Requires [PyTorch](https://pypi.org/project/torch) and [TensorBoard](https://pypi.org/project/tensorboard). No TensorFlow installation required.

## Installation

```sh
pip install tensorboard-reducer
```

## Usage

### Through CLI

```sh
tb-reducer -i 'glob_pattern/of_dirs_to_reduce*' -o output_dir -r mean,std,min,max
```

> **Note**: By default, TensorBoard Reducer expects event files containing identical tags and equal number of steps for all scalars. If e.g. you trained one model for 300 epochs and another for 400 and/or added different tags, see flags `--lax-tags` and `--lax-tags` to remove this restriction.

![Mean of 3 TensorBoard logs](https://raw.githubusercontent.com/janosh/tensorboard-reducer/main/assets/3-runs-mean.png)

`tb-reducer` has the following flags:

- **`-i/--indirs-glob`** (required): Glob pattern of the run directories to reduce.
- **`-o/--outdir`** (required): Name of the directory to save the new reduced run data. If `--format` is `tb-events`, a separate directory will be created for each reduce op (`mean`, `std`, ...) suffixed by the op's name (`outdir-mean`, `outdir-std`, ...). If `--format` is `csv`, a single file will created and `outdir` must be the full file path ending in `.csv`.
- **`-f/--format`** (optional, default: `tb-events`): Output format of reduced TensorBoard runs. Use `tb-events` for writing regular TensorBoard event files or `csv`. If `csv`, `-o/--outdir` must have `.csv` extension and all reduction ops will be written to a single CSV file rather than separate directories for each reduce op. Use `pandas.read_csv("path/to/file.csv", header=[0, 1], index_col=0)` to read data back into memory as a multi-index dataframe.
- **`-r/--reduce-ops`** (optional, default: `mean`): Comma-separated names of numpy reduction ops (`mean`, `std`, `min`, `max`, ...). Default is `mean`. Each reduction is written to a separate `outdir` suffixed by its op name, e.g. if `outdir='my-new-run`, the mean reduction will be written to `my-new-run-mean`.
- **`-w/--overwrite`** (optional, default: `False`): Whether to overwrite existing output directories/CSV files.
- **`--lax-tags`** (optional, default: `False`): Allow different runs have to different sets of tags. In this mode, each tag reduction will run over as many runs as are available for a given tag, even if that's just one. Proceed with caution as not all tags will have the same statistics in downstream analysis.
- **`--lax-steps`** (optional, default: `False`): Allow tags across different runs to have unequal numbers of steps. In this mode, each reduction will only use as many steps as are available in the shortest run (same behavior as `zip(short_list, long_list)`)."

### Through Python API

You can also import `tensorboard_reducer` into a Python script for more complex operations. A simple example that makes use of the full Python API (`load_tb_events`, `reduce_events`, `write_csv`, `write_tb_events`) to get you started:

```py
from tensorboard_reducer import load_tb_events, reduce_events, write_csv, write_tb_events

in_dirs_glob = "glob_pattern/of_directories_to_reduce*"
out_dir = "path/to/output_dir"
out_csv = "path/to/out.csv"
overwrite = False
reduce_ops = ["mean", "min", "max"]

events_dict = load_tb_events(in_dirs_glob)

n_steps, n_events = list(events_dict.values())[0].shape
n_scalars = len(events_dict)

print(
    f"Loaded {n_events} TensorBoard runs with {n_scalars} scalars and {n_steps} steps each"
)
for tag in events_dict.keys():
    print(f" - {tag}")

reduced_events = reduce_events(events_dict, reduce_ops)

for op in reduce_ops:
    print(f"Writing '{op}' reduction to '{out_dir}-{op}'")

write_tb_events(reduced_events, out_dir, overwrite)

write_csv(reduced_events, out_csv, overwrite)
```
