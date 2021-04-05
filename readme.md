![TensorBoard Reducer](https://raw.githubusercontent.com/janosh/tensorboard-reducer/main/assets/tensorboard-reducer.svg)

<h4 align="center">

[![Tests](https://github.com/janosh/tensorboard-reducer/workflows/Tests/badge.svg)](https://github.com/janosh/tensorboard-reducer/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/tensorboard-reducer/main.svg)](https://results.pre-commit.ci/latest/github/janosh/tensorboard-reducer/main)
[![PyPI](https://img.shields.io/pypi/v/tensorboard-reducer)](https://pypi.org/project/tensorboard-reducer)
[![This project supports Python 3.6+](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://python.org/downloads)
[![License](https://img.shields.io/github/license/janosh/tensorboard-reducer?label=License)](/license)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/janosh/tensorboard-reducer?label=Repo+Size)](https://github.com/janosh/tensorboard-reducer/graphs/contributors)

</h4>

Compute reduced statistics (`mean`, `std`, `min`, `max`, `median` or any other [`numpy`](https://numpy.org/doc/stable) operation) of multiple TensorBoard runs matching a directory glob pattern. This can be used after training multiple identical models to reduce the noise in their loss/accuracy/error curves e.g. when trying to establish a statistically significant improvement in training performance.

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
- `-o/--outdir` (required): Name of the directory to save the new reduced run data.
- `-r/--reduce-ops` (required): Comma-separated names of numpy reduction ops (`mean`, `std`, `min`, `max`, ...). Default is `mean`. Each reduction is written to a separate `outdir` suffixed by its op name, e.g. if `outdir='my-new-run`, the mean reduction will be written to `my-new-run-mean`. Only exception is `std` which will create two `outdir`s named `my-new-run-mean+std` and `my-new-run-mean-std`.
- `-f/--format`: Output format of reduced TensorBoard runs. One of `tb-events` for regular TensorBoard event files or `csv`. If `csv`, `-o/--outdir` must have `.csv` extension and all reduction ops will be written to a single CSV file rather than separate directories for each reduce op.
- `-w/--overwrite` (optional): Whether to overwrite existing reduction directories.

## Testing

This project uses [`pytest`](https://docs.pytest.org/en/stable/usage.html). To run the entire test suite:

```sh
python -m pytest
```

To run individual or groups of test files, pass `pytest` a path or glob pattern, respectively:

```sh
python -m pytest tests/test_cumulative.py
python -m pytest **/test_*_metrics.py
```

To run a single test, pass its name to the `-k` flag:

```sh
python -m pytest -k test_precision_recall_curve
```

Consult the [`pytest`](https://docs.pytest.org/en/stable/usage.html) docs for more details.
