from argparse import ArgumentParser
from typing import Dict, List, Optional, Sequence

import numpy as np
from importlib_metadata import version
from numpy.typing import ArrayLike as Array

from .io import load_tb_events, write_csv, write_tb_events


def reduce_events(
    events_dict: Dict[str, Array], reduce_ops: List[str]
) -> Dict[str, Dict[str, Array]]:
    """Perform numpy reduce ops on the last axis of each array
    in a dictionary of scalar TensorBoard event data. Each array enters
    this function with shape (n_timesteps, r_runs) and len(reduce_ops) exit
    with shape (n_timesteps,).

    Args:
        events_dict (dict[str, Array]): Dictionary of arrays to reduce.
        reduce_ops (list[str]): numpy reduce ops.

    Returns:
        dict[str, dict[str, Array]]: Dict of dicts where each subdict holds one reduced array
            for each of the specified reduce ops, e.g. {"loss": {"mean": arr.mean(-1),
            "std": arr.std(-1)}}.
    """

    reductions = {}

    for op in reduce_ops:

        reductions[op] = {}

        for tag, arr in events_dict.items():

            reduce_op = getattr(np, op)

            reductions[op][tag] = reduce_op(arr, axis=-1)

    return reductions


def main(argv: Optional[Sequence[str]] = None) -> int:

    parser = ArgumentParser("TensorBoard Reducer")

    parser.add_argument(
        "-i",
        "--indirs-glob",
        help=(
            "Glob pattern of the run directories to reduce. "
            "Remember to protect wildcards with quotes to prevent shell expansion."
        ),
    )
    parser.add_argument(
        "-o", "--outdir", help="Name of the directory to save the new reduced run data."
    )
    parser.add_argument(
        "-r",
        "--reduce-ops",
        type=lambda s: s.split(","),
        default=["mean"],
        help=(
            "Comma-separated names of numpy reduction ops (mean, std, min, max, ...). Default "
            "is mean. Each reduction is written to a separate outdir suffixed by op name."
        ),
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["tb-events", "csv"],
        default="tb-events",
        help=(
            "Output format of reduced TensorBoard runs. One of `tb-events` for regular "
            "TensorBoard event files or `csv`. If `csv`, `-o/--outdir` must have `.csv` "
            "extension and all reduction ops will be written to a single CSV file rather "
            "than separate directories for each reduce op."
        ),
    )
    parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing reduction directories.",
    )

    tb_version = version("tensorboard_reducer")

    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {tb_version}"
    )
    args = parser.parse_args(argv)

    outdir, indirs_glob = args.outdir, args.indirs_glob
    overwrite, reduce_ops = args.overwrite, args.reduce_ops

    events_dict = load_tb_events(indirs_glob)

    n_steps, n_events = list(events_dict.values())[0].shape
    n_scalars = len(events_dict)

    print(
        f"Loaded {n_events} TensorBoard runs with {n_scalars} scalars and {n_steps} steps each"
    )
    for tag in events_dict.keys():
        print(f" - {tag}")

    reduced_events = reduce_events(events_dict, reduce_ops)

    if args.format == "tb-events":

        for op in reduce_ops:
            print(f"Writing '{op}' reduction to '{outdir}-{op}'")

        write_tb_events(reduced_events, outdir, overwrite)

    elif args.format == "csv":

        print(f"Writing '{reduce_ops}' reductions to '{outdir}'")

        write_csv(reduced_events, outdir, overwrite)


if __name__ == "__main__":
    exit(main())
