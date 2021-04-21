![TensorBoard Reducer](https://raw.githubusercontent.com/janosh/tensorboard-reducer/main/assets/tensorboard-reducer.svg)

<h4 align="center">

[![Tests](https://github.com/janosh/tensorboard-reducer/workflows/Tests/badge.svg)](https://github.com/janosh/tensorboard-reducer/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/tensorboard-reducer/main.svg)](https://results.pre-commit.ci/latest/github/janosh/tensorboard-reducer/main)
[![PyPI](https://img.shields.io/pypi/v/tensorboard-reducer)](https://pypi.org/project/tensorboard-reducer)
[![This project supports Python 3.6+](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://python.org/downloads)
[![License](https://img.shields.io/github/license/janosh/tensorboard-reducer?label=License)](/license)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/janosh/tensorboard-reducer?label=Repo+Size)](https://github.com/janosh/tensorboard-reducer/graphs/contributors)

</h4>

> This project was inspired by [`tensorboard-aggregator`](https://github.com/Spenhouet/tensorboard-aggregator) (similar project built with TensorFlow rather than PyTorch) and [this SO answer](https://stackoverflow.com/a/48774926).

Compute reduced statistics (`mean`, `std`, `min`, `max`, `median` or any other [`numpy`](https://numpy.org/doc/stable) operation) of multiple TensorBoard runs matching a directory glob pattern. This can for instance be used when training multiple identical models to reduce the noise in their loss/accuracy/error curves to establish statistical significance in performance improvements. The aggregation results can be saved to disk either as new TensorBoard event files or in CSV format.

Requires [PyTorch](https://pypi.org/project/torch) and [TensorBoard](https://pypi.org/project/tensorboard). No TensorFlow installation required.

## Installation

```sh
pip install tensorboard-reducer
```

## Usage

Example:

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
