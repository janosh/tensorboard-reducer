from __future__ import annotations

from collections import defaultdict
from typing import Literal, get_args

import pandas as pd
from tqdm import tqdm

from tensorboard_reducer.event_loader import EventAccumulator

HandleDupSteps = Literal["keep-first", "keep-last", "mean", None]


def load_tb_events(
    input_dirs: list[str],
    strict_tags: bool = True,
    strict_steps: bool = True,
    handle_dup_steps: HandleDupSteps | None = None,
    min_runs_per_step: int | None = None,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Read all TensorBoard event files found in input_dirs and return their scalar data
    as a dict with tags as keys (e.g. 'training/loss', 'validation/mae') and 2d arrays
    of shape (n_steps, n_runs) as values.

    Args:
        input_dirs (list[str]): Directory names containing TensorBoard runs to read
            from disk.
        strict_tags (bool, optional): If true, throw error if different runs have
            different sets of tags. Defaults to True.
        strict_steps (bool, optional): If true, throw error if equal tags across
            different runs have unequal numbers of steps. Defaults to True.
        handle_dup_steps (str|None, optional): How to handle duplicate values recorded
            for the same tag and step in a single run directory (can come from multiple
            event files in the same run directory or even from duplicate values in a
            single event file). One of 'keep-first', 'keep-last' or 'mean' which will
            keep the first/last occurrence of duplicate steps and compute their mean,
            respectively. Defaults to None which will raise an error on duplicate steps.
        min_runs_per_step (int|None, optional): Minimum number of runs across which a
            given step must be recorded to be kept. Steps present across less runs are
            dropped. Only plays a role if strict_steps=False. **Warning**: Be aware that
            with this setting, you'll be reducing variable number of runs, however many
            recorded a value for a given step as long as there are at least
            --min-runs-per-step. In other words, the statistics of a reduction will
            change mid-run. Say you're plotting the mean of an error curve, the sample
            size of that mean will drop from 10 down to 4 mid-plot if 4 of your models
            trained for longer than the rest. Be sure to remember when using this.
        verbose (bool, optional): If true, print progress to stdout. Defaults to False.

    Returns:
        dict: A dictionary mapping scalar tags (i.e. keys like 'train/loss', 'val/mae')
            to Pandas DataFrames.
    """
    if not input_dirs:
        msg = f"Expected non-empty list of input directories, got '{input_dirs}'"
        raise ValueError(msg)
    if handle_dup_steps not in (None, "keep-first", "keep-last", "mean"):
        raise ValueError(
            f"unexpected {handle_dup_steps=}, must be one of {get_args(HandleDupSteps)}"
        )

    # Here's where TensorBoard scalars are loaded into memory. Uses a custom
    # EventAccumulator that only loads scalars and ignores histograms, images and other
    # time-consuming data.
    accumulators = [
        EventAccumulator(dirname).Reload()
        for dirname in tqdm(input_dirs, disable=not verbose, desc="Loading runs")
    ]

    # Safety check: make sure all loaded runs have identical tags unless user set
    # strict_tags=False.
    if strict_tags:
        # generate list of scalar tags for all event files
        all_dirs_tags_list = [accumulator.scalar_tags for accumulator in accumulators]

        tags_set = {tag for tags in all_dirs_tags_list for tag in tags}

        missing_tags_report = "".join(
            f"- {in_dir} missing tags: {', '.join(tags_set - {*tags})}\n"
            for in_dir, tags in zip(input_dirs, all_dirs_tags_list)
            if len(tags_set - {*tags}) > 0
        )

        if missing_tags_report:
            raise ValueError(
                f"Some tags are in some logs but not others:\n{missing_tags_report}"
                "\nIf intentional, pass CLI flag --lax-tags or strict_tags=False "
                "to the Python API. With that, each tag reduction uses as many "
                "runs as are available for a given tag, even if that's just one. "
                "Proceed with caution as not all tags will have the same statistics in "
                "downstream analysis."
            )

    load_dict = defaultdict(list)

    for accumulator in tqdm(accumulators, disable=not verbose, desc="Reading tags"):
        in_dir = accumulator.path

        for tag in accumulator.scalar_tags:
            # accumulator.Scalars() returns columns 'step', 'wall_time', 'value'
            df_scalar = pd.DataFrame(accumulator.Scalars(tag)).set_index("step")
            df_scalar = df_scalar.drop(columns="wall_time")

            if handle_dup_steps is None and not df_scalar.index.is_unique:
                raise ValueError(
                    f"Tag '{tag}' from run directory '{in_dir}' contains duplicate "
                    "steps. Please make sure your data wasn't corrupted. If this is "
                    "expected/you want to proceed anyway, specify how to handle "
                    "duplicate values recorded for the same tag and step in a single "
                    "run by passing --handle-dup-steps to the CLI or "
                    "handle_dup_steps='keep-first'|'keep-last'|'mean' to the Python "
                    "API. This will keep the first/last occurrence of duplicate steps "
                    "or take their mean."
                )
            if handle_dup_steps == "mean":
                df_scalar = df_scalar.groupby(df_scalar.index).mean()
            elif handle_dup_steps in ("keep-first", "keep-last"):
                keep = handle_dup_steps.replace("keep-", "")
                df_scalar = df_scalar[~df_scalar.index.duplicated(keep=keep)]

            load_dict[tag].append(df_scalar)

    # Safety check: make sure all loaded runs have equal numbers of steps for each tag
    # unless user set strict_steps=False.
    if strict_steps:
        for tag, lst in load_dict.items():
            n_steps_per_run = [len(df) for df in lst]

            all_runs_equal_steps = n_steps_per_run.count(n_steps_per_run[0]) == len(
                n_steps_per_run
            )

            if not all_runs_equal_steps:
                raise ValueError(
                    f"Unequal number of steps {n_steps_per_run} for different runs for "
                    f"the same tag '{tag}'. If intentional, pass CLI flag --lax-steps "
                    " or strict_steps=False to the Python API. After that, each "
                    "reduction will only use as many steps as are available in the "
                    "shortest run (same behavior as zip())."
                )

    assert len(load_dict) > 0, (
        f"Got {len(input_dirs)} input directories but no TensorBoard event files "
        "found inside them."
    )

    out_dict: dict[str, pd.DataFrame] = {}

    if min_runs_per_step is not None:
        if not isinstance(min_runs_per_step, int) or min_runs_per_step < 1:
            raise ValueError(
                f"Expected positive integer or None, got {min_runs_per_step=}"
            )

        for key, lst in load_dict.items():
            # join='outer' means keep the union of indices from all joined dataframes.
            # That is, we retain all steps as long as any run recorded a value for it.
            # Only makes a difference if strict_steps=False and different runs have
            # non-overlapping steps.
            df_scalar = pd.concat(lst, join="outer", axis=1)
            # count(axis=1) returns the number of non-NaN values in each row
            df_scalar = df_scalar[df_scalar.count(axis=1) >= min_runs_per_step]
            out_dict[key] = df_scalar

    else:
        # join='inner' means keep only the intersection of indices from all joined
        # dataframes. That is, we only retain steps for which all loaded runs recorded
        # a value. Only makes a difference if strict_steps=False and different runs have
        # non-overlapping steps.
        out_dict = {
            key: pd.concat(lst, join="inner", axis=1) for key, lst in load_dict.items()
        }

    if verbose:
        n_tags = len(out_dict)
        if strict_steps and strict_tags:
            n_steps, n_events = next(iter(out_dict.values())).shape
            print(
                f"Loaded {n_events} TensorBoard runs with {n_tags} scalars "
                f"and {n_steps} steps each"
            )
        else:
            print(
                f"Loaded data for {n_tags} tags into arrays of shape (n_steps, n_runs):"
            )

            for tag in list(out_dict)[:50]:
                df_scalar = out_dict[tag]
                print(f"- '{tag}': {df_scalar.shape}")
            if len(out_dict) > 50:
                print("...")

    return out_dict
