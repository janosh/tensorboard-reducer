![TensorBoard Reducer](https://raw.githubusercontent.com/janosh/tensorboard-reducer/main/assets/tensorboard-reducer.svg)

[![Tests](https://github.com/janosh/tensorboard-reducer/actions/workflows/test.yml/badge.svg)](https://github.com/janosh/tensorboard-reducer/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/tensorboard-reducer/main.svg)](https://results.pre-commit.ci/latest/github/janosh/tensorboard-reducer/main)
[![Requires Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/tensorboard-reducer)](https://pypi.org/project/tensorboard-reducer)
[![PyPI Downloads](https://img.shields.io/pypi/dm/tensorboard-reducer)](https://pypistats.org/packages/tensorboard-reducer)

> For a similar project built for TensorFlow rather than PyTorch, see [`tensorboard-aggregator`](https://github.com/Spenhouet/tensorboard-aggregator).

Compute reduced statistics (`mean`, `std`, `min`, `max`, `median` or any other `numpy` operation) of multiple TensorBoard run directories. This can be used e.g. when training multiple identical models (such as deep ensembles) to reduce the noise in their loss/accuracy/error curves and establish statistical significance of performance improvements. Save aggregation results to disk either as new TensorBoard event files, CSV or JSON.

## Installation

```sh
pip install tensorboard-reducer
```

## Usage

### CLI

```sh
tb-reducer runs/of-your-model* -o output-dir -r mean,std,min,max
```

All positional CLI arguments are interpreted as input directories and expected to contain TensorBoard event files. These can be specified individually or with wildcards using shell expansion. You can check you're getting the right input directories by running `echo runs/of-your-model*` before passing them to `tb-reducer`.

**Note**: By default, TensorBoard Reducer expects event files to contain identical tags and equal number of steps for all scalars. If you trained one model for 300 epochs and another for 400 and/or recorded different sets of metrics (tags in TensorBoard lingo) for each of them, see CLI flags `--lax-steps` and `--lax-tags` to disable this safeguard.

![Mean of 3 TensorBoard logs](https://raw.githubusercontent.com/janosh/tensorboard-reducer/main/assets/3-runs-mean.png)

In addition, `tb-reducer` has the following flags:

- **`-o/--outpath`** (required): File or directory where to save output on disk. Will save as a CSV file if path ends in '.csv' extension or else as TensorBoard run directories, one for each reduction suffixed by the operation's name, e.g. `'outpath-mean'`, `'outpath-max'`, etc. If output format is CSV, a single file will be created with two-level header containing one column for each combination of tag and reduce operation. Tag names will be in top-level header, reduce op in second level. **Hint**: Use `pandas.read_csv("path/to/file.csv", header=[0, 1], index_col=0)` to read CSV data back into a multi-index dataframe.
- **`-r/--reduce-ops`** (optional, default: `mean`): Comma-separated names of numpy reduction ops (`mean`, `std`, `min`, `max`, ...). Each reduction is written to a separate `outpath` suffixed by its op name. E.g. if `outpath='reduced-run'`, the mean reduction will be written to `'reduced-run-mean'`.
- **`-f/--overwrite`** (optional, default: `False`): Whether to overwrite existing output directories/CSV files. For safety, the overwrite operation will abort with an error if the file/directory to overwrite is not a CSV and does not look like a TensorBoard run directory (i.e. does not start with `'events.out'`).
- **`--lax-tags`** (optional, default: `False`): Allow different runs have to different sets of tags. In this mode, each tag reduction will run over as many runs as are available for a given tag, even if that's just one. Proceed with caution as not all tags will have the same statistics in downstream analysis.
- **`--lax-steps`** (optional, default: `False`): Allow tags across different runs to have unequal numbers of steps. In this mode, each reduction will only use as many steps as are available in the shortest run (same behavior as `zip(short_list, long_list)` which stops when `short_list` is exhausted).
- **`--handle-dup-steps`** (optional, default: `None`): How to handle duplicate values recorded for the same tag and step in a single run. One of `'keep-first'`, `'keep-last'`, `'mean'`. `'keep-first/last'` will keep the first/last occurrence of duplicate steps while 'mean' computes their mean. Default behavior is to raise `AssertionError` on duplicate steps.
- **`--min-runs-per-step`** (optional, default: `None`): Minimum number of runs across which a given step must be recorded to be kept. Steps present across less runs are dropped. Only plays a role if `lax_steps` is true. **Warning**: Be aware that with this setting, you'll be reducing variable number of runs, however many recorded a value for a given step as long as there are at least `--min-runs-per-step`. In other words, the statistics of a reduction will change mid-run. Say you're plotting the mean of an error curve, the sample size of that mean will drop from, say, 10 down to 4 mid-plot if 4 of your models trained for longer than the rest. Be sure to remember when using this.
- **`-v/--version`** (optional): Get the current version.

### Python API

You can also import `tensorboard_reducer` into a Python script for more complex operations. A simple example that makes use of the full Python API (`load_tb_events`, `reduce_events`, `write_csv`, `write_tb_events`) to get you started:

```py
from glob import glob

import tensorboard_reducer as tbr

in_dirs = glob("glob_pattern/of_directories_to_reduce*")
events_out_dir = "path/to/output_dir"
csv_out_path = "path/to/out.csv"
overwrite = False
reduce_ops = ("mean", "min", "max")

events_dict = tbr.load_tb_events(in_dirs)

n_scalars = len(events_dict)
n_steps, n_events = list(events_dict.values())[0].shape

print(
    f"Loaded {n_events} TensorBoard runs with {n_scalars} scalars and {n_steps} steps each"
)
print(", ".join(events_dict))

reduced_events = tbr.reduce_events(events_dict, reduce_ops)

for op in reduce_ops:
    print(f"Writing '{op}' reduction to '{events_out_dir}-{op}'")

tbr.write_tb_events(reduced_events, events_out_dir, overwrite)

print(f"Writing results to '{csv_out_path}'")

tbr.write_df(reduced_events, csv_out_path, overwrite)

print("Reduction complete")
```
