from __future__ import annotations

from argparse import ArgumentParser
from importlib.metadata import version

import pandas as pd

from .load import load_tb_events
from .write import write_data_file, write_tb_events


def reduce_events(
    events_dict: dict[str, pd.DataFrame], reduce_ops: list[str]
) -> dict[str, dict[str, pd.DataFrame]]:
    """Perform numpy reduce ops on the last axis of each array in a dictionary of
    scalar TensorBoard event data. Each array enters
    this function with shape (n_timesteps, r_runs) and len(reduce_ops) exit
    with shape (n_timesteps,).

    Args:
        events_dict (dict[str, pd.DataFrame]): Dict of arrays to reduce.
        reduce_ops (list[str]): numpy reduce ops.

    Returns:
        dict[str, dict[str, pd.DataFrame]]: Dict of dicts where each subdict holds one
            reduced array for each of the specified reduce ops, e.g.
            {"loss": {"mean": arr.mean(-1), "std": arr.std(-1)}}.
    """
    reductions: dict[str, dict[str, pd.DataFrame]] = {}

    for op in reduce_ops:

        reductions[op] = {}

        for tag, df in events_dict.items():

            reductions[op][tag] = getattr(df, op)(axis=1)

    return reductions


def main(argv: list[str] = None) -> int:
    """Implement tb-reducer CLI.

    Args:
        argv (list[str], optional): Command line arguments. Defaults to None.

    Returns:
        int: 0 if successful else error code
    """
    parser = ArgumentParser(
        "TensorBoard Reducer",
        description="Compute reduced statistics (mean, std, min, max, median, etc.) of "
        "multiple TensorBoard runs matching a directory glob pattern.",
    )

    parser.add_argument(
        "input_dirs",
        nargs="+",
        help=(
            "List of run directories to reduce. Use shell expansion (e.g. "
            "runs/of_some_model/*) to glob as many directories as required."
        ),
    )
    parser.add_argument(
        "-o",
        "--outpath",
        help=(
            "File or directory where to save output on disk. Will save as a CSV file "
            "if path ends in '.csv' extension or else as TensorBoard run directories, "
            "one for each reduce op suffixed by the op's name, e.g. 'outpath-mean', "
            "'outpath-max', etc. If output format is CSV, the output file will have a "
            "two-level header containing one column for each combination of tag and "
            "reduce operation with tag name in first and reduce op in second level."
        ),
    )
    parser.add_argument(
        "-r",
        "--reduce-ops",
        type=lambda s: s.split(","),
        default=["mean"],
        help="Comma-separated names of numpy reduction ops (mean, std, min, max, ...). "
        "Default is mean. Each reduction is written to a separate output directory "
        "suffixed by op name. E.g. if outpath='reduced-run', the mean reduction will "
        "be written to 'reduced-run-mean'.",
    )
    parser.add_argument(
        "-f",
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing reduction directories.",
    )
    parser.add_argument(
        "--lax-tags",
        action="store_true",
        help="Don't error if different runs have different sets of tags.",
    )
    parser.add_argument(
        "--lax-steps",
        action="store_true",
        help="Don't error if equal tags across different runs have unequal numbers of "
        "steps.",
    )
    parser.add_argument(
        "--handle-dup-steps",
        choices=("keep-first", "keep-last", "mean"),
        default=None,
        help="How to handle duplicate values recorded for the same tag and step in a "
        "single run. 'keep-first/last' will keep the first/last occurrence of "
        "duplicate steps while 'mean' compute their mean. Default behavior is to raise "
        "an error on duplicate steps.",
    )
    parser.add_argument(
        "--min-runs-per-step",
        type=int,
        default=None,
        help="Minimum number of runs across which a given step must be recorded to be "
        "kept. Steps present across less runs are dropped. Only plays a role if "
        "strict_steps=False. Warning: Be aware with this setting, you'll be reducing "
        "variable number of runs, however many recorded a value for a given step as "
        "long as there are at least --min-runs-per-step. That is, the statistics of a "
        "reduction will change mid-run. Say you're plotting the mean of an error "
        "curve, the sample size of that mean will drop from, say, 10 down to 4 "
        "mid-plot if 4 of your models trained for longer than the rest. Be sure to "
        "remember when using this.",
    )

    tb_version = version("tensorboard_reducer")

    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s v{tb_version}"
    )
    args = parser.parse_args(argv)

    outpath, overwrite, reduce_ops = args.outpath, args.overwrite, args.reduce_ops

    events_dict = load_tb_events(
        args.input_dirs,
        strict_tags=not args.lax_tags,
        strict_steps=not args.lax_steps,
        handle_dup_steps=args.handle_dup_steps,
        min_runs_per_step=args.min_runs_per_step,
    )

    n_scalars = len(events_dict)

    if not args.lax_steps and not args.lax_tags:
        n_steps, n_events = list(events_dict.values())[0].shape

        print(
            f"Loaded {n_events} TensorBoard runs with {n_scalars} scalars "
            f"and {n_steps} steps each"
        )
        if n_scalars < 20:
            print(", ".join(events_dict))
    elif n_scalars < 20:
        print(
            "Loaded data for the following tags into arrays of shape (n_steps, n_runs):"
        )
        for tag, df in events_dict.items():
            print(f"- '{tag}': {df.shape}")

    reduced_events = reduce_events(events_dict, reduce_ops)

    if outpath.endswith(".csv"):

        write_data_file(reduced_events, outpath, overwrite)

        print(f"Wrote {', '.join(reduce_ops)} reductions to '{outpath}'")

    else:

        write_tb_events(reduced_events, outpath, overwrite)

        for op in reduce_ops:
            print(f"Wrote '{op}' reduction to '{outpath}-{op}'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
