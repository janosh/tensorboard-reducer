from __future__ import annotations

import os
from typing import Any

import pandas as pd

_known_extensions = (".csv", ".json", ".xls", ".xlsx")


def _rm_rf_or_raise(path: str, overwrite: bool) -> None:
    """Remove the directory tree below dir if overwrite is True.

    Args:
        path (str): The directory whose children will be removed if overwrite=True.
        overwrite (bool): Whether to overwrite existing.

    Raises:
        FileExistsError: If path exists and overwrite=False.
    """
    if os.path.exists(path):  # True if dir is either file or directory

        # for safety, check dir is either TensorBoard run or CSV file
        # to make it harder to delete files not created by this program
        is_tb_dir = os.path.isdir(path) and all(
            x.startswith("events.out") for x in os.listdir(path)
        )
        # use `ext in path` instead of endswith() to handle compressed files
        # (.csv.gz, .json.bz2, etc.)
        is_data_file = any(ext in path.lower() for ext in _known_extensions)

        if overwrite and (is_data_file or is_tb_dir):
            os.system(f"rm -rf {path}")
        elif overwrite:
            ValueError(
                f"Received the overwrite flag but the content of '{path}' does not "
                "look like it was written by this program. Please make sure you really "
                f"want to delete '{path}' and then do so manually."
            )
        else:
            raise FileExistsError(
                f"'{path}' already exists, pass overwrite=True"
                " (-f/--overwrite in CLI) to proceed anyway"
            )


def write_tb_events(
    data_to_write: dict[str, dict[str, pd.DataFrame]],
    out_dir: str,
    overwrite: bool = False,
) -> list[str]:
    """Writes a dictionary with tags as keys and reduced TensorBoard scalar data as
    values to disk as a new TensorBoard event file in a newly created or overwritten
    `out_dir` directory (depending on `overwrite`).

    Inspired by https://stackoverflow.com/a/48774926.

    Args:
        data_to_write (dict[str, dict[str, pd.DataFrame]]): Data to write to disk.
            Assumes 1st-level keys are reduce ops (mean, std, ...) and 2nd-level are
            TensorBoard tags.
        out_dir (str): Name of the directory to save the new reduced run data. Will
            have the reduce op name (e.g. '-mean'/'-std') appended.
        overwrite (bool): Whether to overwrite existing reduction directories.
            Defaults to False.

    Returns:
        list[str]: List of paths to the new TensorBoard event files.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorflow.summary import SummaryWriter
        except ImportError:
            raise ImportError(
                "Cannot import SummaryWriter from torch nor tensorflow."
                "Install either to create new TensorBoard event files."
            )
    out_dirs: list[str] = []
    data_to_write = data_to_write.copy()  # make copy since we modify std data in place

    out_dir_op_connector = "" if out_dir.endswith(("/", "\\")) else "-"

    # handle std reduction separately as we use writer.add_scalars to write mean +/- std
    if {"mean", "std"}.issubset(data_to_write):

        mean_dict = data_to_write["mean"]
        # remove std from data_to_write so we don't write it twice
        std_dict = data_to_write.pop("std")

        for sign, symbol in ((1, "+"), (-1, "-")):
            std_out_dir = f"{out_dir}{out_dir_op_connector}mean{symbol}std"

            _rm_rf_or_raise(std_out_dir, overwrite)
            out_dirs.append(std_out_dir)

            writer = SummaryWriter(std_out_dir)

            for (tag, means), stds in zip(mean_dict.items(), std_dict.values()):
                # we can safely zip(means, stds): they have the same length and same
                # step values because the same data went into both reductions
                for (step, mean), std in zip(means.items(), stds.to_numpy()):
                    writer.add_scalar(tag, mean + sign * std, step)

        writer.close()

    # loop over each reduce operation (e.g. mean, min, max, median)
    for op, events_dict in data_to_write.items():

        op_out_dir = f"{out_dir}{out_dir_op_connector}{op}"
        out_dirs.append(op_out_dir)

        _rm_rf_or_raise(op_out_dir, overwrite)

        writer = SummaryWriter(op_out_dir)

        for tag, series in events_dict.items():
            for step, value in series.items():
                writer.add_scalar(tag, value, step)

        # Important for allowing write_events() to overwrite. Without it,
        # try_rmtree will raise OSError: [Errno 16] Device or resource busy
        # trying to delete the existing out_dir.
        writer.close()

    return out_dirs


def write_df(*args: Any) -> None:
    """Inform users of breaking change if they try to use the old API."""
    raise NotImplementedError(
        "write_df() was renamed to write_data_file() in tensorboard-reducer v0.2.8"
    )


def write_data_file(
    data_to_write: dict[str, dict[str, pd.DataFrame]],
    out_path: str,
    overwrite: bool = False,
) -> None:
    """Writes reduced TensorBoard data passed as dict of dicts to a CSV file.

    Use `pandas.read_csv("path/to/file.csv", header=[0, 1], index_col=0)` to read CSV
    data back into a multi-index dataframe.

    Args:
        data_to_write (dict[str, dict[str, pd.DataFrame]]): Data to write to disk.
            Assumes 1st-level keys are reduce ops (mean, std, ...) and 2nd-level are
            TensorBoard tags.
        out_path (str): CSV, JSON or Excel file path where the reduced data will be
            written. Supports all compression formats that Pandas supports. Simply
            change the file extension. For example .csv.gz, .csv.gzip, .json.bz2, etc.
        overwrite (bool): Whether to overwrite existing reduction directories.
            Defaults to False.
    """
    _rm_rf_or_raise(out_path, overwrite)

    # create multi-index dataframe from event data with reduce op names as 1st-level col
    # names and tag names as 2nd level
    dict_of_dfs = {op: pd.DataFrame(dic) for op, dic in data_to_write.items()}
    df = pd.concat(dict_of_dfs, axis=1)
    df.columns = df.columns.swaplevel(0, 1)
    df.index.name = "step"

    # let pandas handle compression inference from extensions (.csv.gz, .json.bz2, etc.)
    basename = os.path.basename(out_path)
    if ".csv" in basename.lower():
        df.to_csv(out_path)
    elif ".json" in basename.lower():
        df.to_json(out_path)
    elif ".xls" in out_path.lower():
        df.to_excel(out_path)
    else:
        raise ValueError(
            f"{out_path=} has unknown extension, should be one of {_known_extensions} "
            " or compressed versions thereof like '.csv.gz', '.json.bz2', etc."
        )
