from argparse import ArgumentParser
from importlib.metadata import version
from typing import Dict, List, Optional, Sequence

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

        for tag, df in events_dict.items():

            reductions[op][tag] = getattr(df, op)(axis=1)

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
    parser.add_argument(
        "--lax-tags",
        action="store_false",
        help="Don't error if equal tags across different runs have unequal numbers of steps.",
    )
    parser.add_argument(
        "--lax-steps",
        action="store_false",
        help="Don't error if different runs have different sets of tags.",
    )

    tb_version = version("tensorboard_reducer")

    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {tb_version}"
    )
    args = parser.parse_args(argv)

    outdir, overwrite, reduce_ops = args.outdir, args.overwrite, args.reduce_ops

    events_dict = load_tb_events(
        args.indirs_glob, strict_tags=args.lax_tags, strict_steps=args.lax_steps
    )

    n_scalars = len(events_dict)

    if not args.lax_steps and not args.lax_tags:
        n_steps, n_events = list(events_dict.values())[0].shape

        print(
            f"Loaded {n_events} TensorBoard runs with {n_scalars} scalars "
            f"and {n_steps} steps each"
        )
        if n_scalars < 20:
            print(", ".join(events_dict.keys()))
    elif n_scalars < 20:
        print("Loaded the following tags and step counts:")
        for tag, lst in events_dict.items():
            print(f"- {tag}: {[len(arr) for arr in lst]}")

    reduced_events = reduce_events(events_dict, reduce_ops)

    if args.format == "tb-events":

        write_tb_events(reduced_events, outdir, overwrite)

        for op in reduce_ops:
            print(f"Wrote '{op}' reduction to '{outdir}-{op}'")

    elif args.format == "csv":

        write_csv(reduced_events, outdir, overwrite)

        print(f"Wrote '{reduce_ops}' reductions to '{outdir}'")

    else:
        raise ValueError(
            f"unexpected output format '{args.format}', chose one of 'tb-events'|'csv'"
        )


if __name__ == "__main__":
    exit(main())
