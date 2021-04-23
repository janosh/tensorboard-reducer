import os
from glob import glob
from typing import Dict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike as Array
from torch.utils.tensorboard import SummaryWriter

from .event_loader import EventAccumulator


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

    indirs = glob(indirs_glob)
    assert len(indirs) > 0, f"No runs found for glob pattern '{indirs_glob}'"

    accumulators = [EventAccumulator(dirname).Reload() for dirname in indirs]

    tags = accumulators[0].Tags()["scalars"]

    for accumulator, indir in zip(accumulators, indirs):
        # assert all runs have the same tags for scalar data
        tags_i = accumulator.Tags()["scalars"]
        assert tags == tags_i, (
            f"mismatching tags between one or more input dirs: {tags} "
            f"from '{indirs[0]}' != {tags_i} from '{indir}'"
        )

    out_dict = {t: [] for t in tags}

    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in accumulators]):
            out_dict[tag].append([e.value for e in events])

    return {key: np.array(val) for key, val in out_dict.items()}


def force_rm_or_raise(path: str, overwrite: bool) -> None:
    """Remove the directory tree below dir if overwrite is True.

    Args:
        dir (str): The directory whose children will be removed if overwrite.
        overwrite (bool): Whether to overwrite existing.

    Raises:
        FileExistsError: If dir exists and not overwrite.
    """
    if os.path.exists(path):  # True if dir is either file or directory

        # for safety, check dir is either TensorBoard run or CSV file
        # to make it harder to delete files not created by this program
        is_csv_file = path.endswith(".csv")
        is_tb_run_dir = os.path.isdir(path) and os.listdir(path)[0].startswith(
            "events.out"
        )

        print(f"{overwrite=}")
        if overwrite and (is_csv_file or is_tb_run_dir):
            os.system(f"rm -rf {path}")
        elif overwrite:
            ValueError(
                f"Received the overwrite flag but the content of '{path}' does not look like"
                " it was writtin by this program. Please make sure you really want to delete"
                " that and then do so manually."
            )
        else:
            raise FileExistsError(
                f"'{path}' already exists, pass overwrite=True"
                " (-w/--overwrite in CLI) to proceed anyway"
            )


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

    # handle std reduction separately as we use writer.add_scalars to write mean +/- std
    if all(x in data_to_write.keys() for x in ["mean", "std"]):

        std_dict = data_to_write.pop("std")
        mean_dict = data_to_write["mean"]

        std_dir = f"{outdir}-std"

        force_rm_or_raise(std_dir, overwrite)

        writer = SummaryWriter(std_dir)

        for (tag, means), stds in zip(mean_dict.items(), std_dict.values()):
            for idx, (mean, std) in enumerate(zip(means, stds)):
                writer.add_scalars(
                    tag, {"mean+std": mean + std, "mean-std": mean - std}, idx
                )

        writer.close()

    # loop over each reduce operation (e.g. mean, min, max, median)
    for op, events_dict in data_to_write.items():

        op_outdir = f"{outdir}-{op}"

        force_rm_or_raise(op_outdir, overwrite)

        writer = SummaryWriter(op_outdir)

        for tag, data in events_dict.items():
            for idx, scalar in enumerate(data):
                writer.add_scalar(tag, scalar, idx)

        # Important for allowing write_events() to overwrite. Without it,
        # try_rmtree will raise OSError: [Errno 16] Device or resource busy
        # trying to delete the existing outdir.
        writer.close()


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

    assert csv_path.endswith(".csv"), f"{csv_path=} should have a .csv extension"

    force_rm_or_raise(csv_path, overwrite)

    # create multi-index dataframe from event data with reduce op names as 1st-level col
    # names and tag names as 2nd level
    dict_of_dfs = {op: pd.DataFrame(dic) for op, dic in data_to_write.items()}
    df = pd.concat(dict_of_dfs, axis=1)
    df.columns = df.columns.swaplevel(0, 1)

    df.index.name = "timestep"
    df.to_csv(csv_path)
