import os
from collections import defaultdict
from glob import glob
from typing import Dict, Union

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from .event_loader import EventAccumulator


def load_tb_events(
    indirs_glob: str,
    strict_tags: bool = True,
    strict_steps: bool = True,
    handle_dup_steps: Union[str, None] = None,
    min_runs_per_step: Union[int, None] = None,
) -> Dict[str, pd.DataFrame]:
    """Read the TensorBoard event files matching the provided glob pattern
    and return their scalar data as a dict with tags ('training/loss',
    'validation/mae', etc.) as keys and 2d arrays of shape (n_timesteps, r_runs)
    as values.

    Args:
        indirs_glob (str): Glob pattern of the run directories to read from disk.
        strict_tags (bool, optional): If true, throw error if different runs have different
            sets of tags. Defaults to True.
        strict_steps (bool, optional): If true, throw error if equal tags across different
            runs have unequal numbers of steps. Defaults to True.
        handle_dup_steps (str|None, optional): How to handle duplicate values recorded for the
            same tag and step in a single run directory (can come from multiple event files in
            the same run directory or even from duplicate values in a single event file). One
            of 'keep-first', 'keep-last' or 'mean' which will keep the first/last occurrence of
            duplicate steps and compute their mean, respectively. Defaults to None which will
            raise an error on duplicate steps.
        min_runs_per_step (int|None, optional): Minimum number of runs across which a given
            step must be recorded to be kept. Steps present across less runs are dropped. Only
            plays a role if strict_steps=False. **Warning**: Be aware that with this setting,
            you'll be reducing variable number of runs, however many recorded a value for a
            given step as long as there are at least --min-runs-per-step. In other words,
            the statistics of a reduction will change mid-run. Say you're plotting the mean of
            an error curve, the sample size of that mean will drop from, say, 10 down to 4
            mid-plot if 4 of your models trained for longer than the rest. Be sure to remember
            when using this.


    Returns:
        dict: A dictionary mapping scalar tags (i.e. keys like 'train/loss', 'val/mae') to
            pandas DataFrames.
    """

    indirs = glob(indirs_glob)
    assert len(indirs) > 0, f"No runs found for glob pattern '{indirs_glob}'"

    # Here's where TensorBoard scalars are loaded into memory. Uses a custom EventAccumulator
    # that only loads scalars, ignores histograms, images and other time-consuming data.
    accumulators = [EventAccumulator(dirname).Reload() for dirname in indirs]

    # Safety check: make sure all loaded runs have identical tags unless user chose to ignore.
    if strict_tags:
        # generate list of scalar tags for all event files each in alphabetical order
        all_dirs_tags_list = sorted(
            accumulator.Tags()["scalars"] for accumulator in accumulators
        )
        first_tags = all_dirs_tags_list[0]

        all_runs_same_tags = all(first_tags == tags for tags in all_dirs_tags_list)

        tags_set = {tag for tags in all_dirs_tags_list for tag in tags}

        missing_tags_report = "".join(
            f"- {dir} missing tags: {', '.join(tags_set - {*tags})}\n"
            for dir, tags in zip(indirs, all_dirs_tags_list)
        )

        assert all_runs_same_tags, (
            f"Some tags appear in some event files but not others:\n{missing_tags_report}\nIf "
            "this is intentional, pass --lax-tags to the CLI or strict_tags=False to the "
            "Python API. After that, each tag reduction will run over as many runs as are "
            "available for a given tag, even if that's just one. Proceed with caution "
            "as not all tags will have the same statistics in downstream analysis."
        )

    out_dict = defaultdict(list)

    for indir, accumulator in zip(indirs, accumulators):
        tags = accumulator.Tags()["scalars"]

        for tag in tags:
            # dataframes use 'step' as index leaving 'wall_time' and 'value' as cols
            df = pd.DataFrame(accumulator.Scalars(tag)).set_index("step")
            df = df.drop(columns="wall_time")

            if handle_dup_steps is None:
                assert df.index.is_unique, (
                    f"Tag '{tag}' from run directory '{indir}' contains duplicate steps. "
                    "Please make sure your data wasn't corrupted. If this is expected/you "
                    "want to proceed anyway, specify how to handle duplicate values recorded "
                    "for the same tag and step in a single run by passing --handle-dup-steps "
                    "to the CLI or handle_dup_steps='keep-first'|'keep-last'|'mean' to the "
                    "Python API. This will keep the first/last occurrence of duplicate steps "
                    "or take their mean."
                )
            elif handle_dup_steps == "mean":
                df = df.groupby(df.index).mean()
            elif handle_dup_steps in ["keep-first", "keep-last"]:
                keep = handle_dup_steps.replace("keep-", "")
                df = df[~df.index.duplicated(keep=keep)]
            else:
                raise ValueError(
                    f"unexpected value for {handle_dup_steps=}, should be one of "
                    "'first', 'last', 'mean', None."
                )

            out_dict[tag].append(df)

    # Safety check: make sure all loaded runs have equal numbers of steps for each tag unless
    # user chose to ignore.
    if strict_steps:
        for tag, lst in out_dict.items():
            n_steps_per_run = [len(df) for df in lst]

            all_runs_equal_steps = n_steps_per_run.count(n_steps_per_run[0]) == len(
                n_steps_per_run
            )

            assert all_runs_equal_steps, (
                f"Unequal number of steps {n_steps_per_run} across different runs for the "
                f"same tag '{tag}'. If this is intentional, pass --lax-steps "
                "to the CLI or strict_steps=False when using the Python API. After that, each "
                "reduction will only use as many steps as are available in the shortest run "
                "(same behavior as zip())."
            )

    assert len(out_dict) > 0, (
        f"Glob pattern '{indirs_glob}' matched {len(indirs)} directories but no TensorBoard "
        "event files found inside them."
    )

    if min_runs_per_step is not None:
        assert (
            type(min_runs_per_step) == int and min_runs_per_step > 0
        ), f"got {min_runs_per_step=}, expected positive integer"

        for key, lst in out_dict.items():
            # join='outer' means keep the union of indices from all joined dataframes. That is,
            # we retain all steps as long as any run recorded a value for it. Only makes a
            # difference if strict_steps=False and different runs have non-overlapping steps.
            df = pd.concat(lst, join="outer", axis=1)
            # count(axis=1) returns the number of non-NaN values in each row
            df = df[df.count(axis=1) >= min_runs_per_step]
            out_dict[key] = df

        return out_dict
    else:
        # join='inner' means keep only the intersection of indices from all joined dataframes.
        # That is, we only retain steps for which all loaded runs recorded a value. Only makes
        # a difference if strict_steps=False and different runs have non-overlapping steps.
        return {
            key: pd.concat(lst, join="inner", axis=1) for key, lst in out_dict.items()
        }


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
                " (-f/--overwrite in CLI) to proceed anyway"
            )


def write_tb_events(
    data_to_write: Dict[str, Dict[str, pd.DataFrame]],
    outdir: str,
    overwrite: bool = False,
) -> None:
    """Writes data in dict to disk as TensorBoard event files in a newly created/overwritten
    outdir directory.

    Inspired by https://stackoverflow.com/a/48774926.

    Args:
        data_to_write (dict[str, dict[str, pd.DataFrame]]): Data to write to disk. Assumes
            1st-level keys are reduce ops (mean, std, ...) and 2nd-level are TensorBoard tags.
        outdir (str): Name of the directory to save the new reduced run data. Will
            have the reduce op name (e.g. '-mean'/'-std') appended.
        overwrite (bool): Whether to overwrite existing reduction directories.
            Defaults to False.
    """

    # handle std reduction separately as we use writer.add_scalars to write mean +/- std
    if "mean" in data_to_write.keys() and "std" in data_to_write.keys():

        std_dict = data_to_write.pop("std")
        mean_dict = data_to_write["mean"]

        std_dir = f"{outdir}-std"

        force_rm_or_raise(std_dir, overwrite)

        writer = SummaryWriter(std_dir)

        for (tag, means), stds in zip(mean_dict.items(), std_dict.values()):
            # we can safely assume mean and std will the same length and same step values
            # as the same data went into both reductions
            for (step, mean), std in zip(means.items(), stds.to_numpy()):
                writer.add_scalars(
                    tag, {"mean+std": mean + std, "mean-std": mean - std}, step
                )

        writer.close()

    # loop over each reduce operation (e.g. mean, min, max, median)
    for op, events_dict in data_to_write.items():

        op_outdir = f"{outdir}-{op}"

        force_rm_or_raise(op_outdir, overwrite)

        writer = SummaryWriter(op_outdir)

        for tag, series in events_dict.items():
            for step, value in series.items():
                writer.add_scalar(tag, value, step)

        # Important for allowing write_events() to overwrite. Without it,
        # try_rmtree will raise OSError: [Errno 16] Device or resource busy
        # trying to delete the existing outdir.
        writer.close()


def write_csv(
    data_to_write: Dict[str, Dict[str, pd.DataFrame]],
    csv_path: str,
    overwrite: bool = False,
) -> None:
    """Writes reduced TensorBoard data passed as dict of dicts (1st arg) to a CSV file
    path (2nd arg).

    Use `pandas.read_csv("path/to/file.csv", header=[0, 1], index_col=0)` to read CSV data
    back into a multi-index dataframe.

    Args:
        data_to_write (dict[str, dict[str, pd.DataFrame]]): Data to write to disk. Assumes
            1st-level keys are reduce ops (mean, std, ...) and 2nd-level are TensorBoard tags.
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

    df.index.name = "step"
    df.to_csv(csv_path)
