from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .event_loader import EventAccumulator


def load_tb_events(
    input_dirs: list[str],
    strict_tags: bool = True,
    strict_steps: bool = True,
    handle_dup_steps: str | None = None,
    min_runs_per_step: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Read all TensorBoard event files found in input_dirs and return their scalar data
    as a dict with tags as keys (e.g. 'training/loss', 'validation/mae') and 2d arrays
    of shape (n_timesteps, r_runs) as values.

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

    Returns:
        dict: A dictionary mapping scalar tags (i.e. keys like 'train/loss', 'val/mae')
            to Pandas DataFrames.
    """
    assert (
        len(input_dirs) > 0
    ), f"Expected non-empty list of input directories, got '{input_dirs}'"

    # Here's where TensorBoard scalars are loaded into memory. Uses a custom
    # EventAccumulator that only loads scalars and ignores histograms, images and other
    # time-consuming data.
    accumulators = [EventAccumulator(dirname).Reload() for dirname in input_dirs]

    # Safety check: make sure all loaded runs have identical tags unless user set
    # strict_tags=False.
    if strict_tags:
        # generate list of scalar tags for all event files each in alphabetical order
        all_dirs_tags_list = sorted(
            accumulator.scalar_tags for accumulator in accumulators
        )
        first_tags = all_dirs_tags_list[0]

        all_runs_same_tags = all(first_tags == tags for tags in all_dirs_tags_list)

        tags_set = {tag for tags in all_dirs_tags_list for tag in tags}

        missing_tags_report = "".join(
            f"- {dir} missing tags: {', '.join(tags_set - {*tags})}\n"
            for dir, tags in zip(input_dirs, all_dirs_tags_list)
            if len(tags_set - {*tags}) > 0
        )

        assert all_runs_same_tags, (
            f"Some tags appear only in some logs but not others:\n{missing_tags_report}"
            "\nIf this is intentional, pass --lax-tags to the CLI or strict_tags=False "
            "to the Python API. After that, each tag reduction will run over as many "
            "runs as are available for a given tag, even if that's just one. Proceed "
            "with caution as not all tags will have the same statistics in downstream "
            "analysis."
        )

    load_dict = defaultdict(list)

    for indir, accumulator in zip(input_dirs, accumulators):
        tags = accumulator.scalar_tags

        for tag in tags:
            # dataframes use 'step' as index leaving 'wall_time' and 'value' as cols
            df = pd.DataFrame(accumulator.Scalars(tag)).set_index("step")
            df = df.drop(columns="wall_time")

            if handle_dup_steps is None:
                assert df.index.is_unique, (
                    f"Tag '{tag}' from run directory '{indir}' contains duplicate "
                    "steps. Please make sure your data wasn't corrupted. If this is "
                    "expected/you want to proceed anyway, specify how to handle "
                    "duplicate values recorded for the same tag and step in a single "
                    "run by passing --handle-dup-steps to the CLI or "
                    "handle_dup_steps='keep-first'|'keep-last'|'mean' to the Python "
                    "API. This will keep the first/last occurrence of duplicate steps "
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

            load_dict[tag].append(df)

    # Safety check: make sure all loaded runs have equal numbers of steps for each tag
    # unless user set strict_steps=False.
    if strict_steps:
        for tag, lst in load_dict.items():
            n_steps_per_run = [len(df) for df in lst]

            all_runs_equal_steps = n_steps_per_run.count(n_steps_per_run[0]) == len(
                n_steps_per_run
            )

            assert all_runs_equal_steps, (
                f"Unequal number of steps {n_steps_per_run} across different runs for "
                f"the same tag '{tag}'. If this is intentional, pass --lax-steps to "
                "the CLI or strict_steps=False when using the Python API. After that, "
                "each reduction will only use as many steps as are available in the "
                "shortest run (same behavior as zip())."
            )

    assert len(load_dict) > 0, (
        f"Got {len(input_dirs)} input directories but no TensorBoard event files "
        "found inside them."
    )

    out_dict: dict[str, pd.DataFrame] = {}

    if min_runs_per_step is not None:
        assert (
            type(min_runs_per_step) == int and min_runs_per_step > 0
        ), f"got {min_runs_per_step=}, expected positive integer"

        for key, lst in load_dict.items():
            # join='outer' means keep the union of indices from all joined dataframes.
            # That is, we retain all steps as long as any run recorded a value for it.
            # Only makes a difference if strict_steps=False and different runs have
            # non-overlapping steps.
            df = pd.concat(lst, join="outer", axis=1)
            # count(axis=1) returns the number of non-NaN values in each row
            df = df[df.count(axis=1) >= min_runs_per_step]
            out_dict[key] = df

        return out_dict
    else:
        # join='inner' means keep only the intersection of indices from all joined
        # dataframes. That is, we only retain steps for which all loaded runs recorded
        # a value. Only makes a difference if strict_steps=False and different runs have
        # non-overlapping steps.
        return {
            key: pd.concat(lst, join="inner", axis=1) for key, lst in load_dict.items()
        }
