from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import pandas as pd


def reduce_events(
    events_dict: dict[str, pd.DataFrame],
    reduce_ops: Sequence[str],
    *,
    verbose: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Perform numpy reduce operations along the last dimension of each array in a
    dictionary of scalar TensorBoard event data. Each array (1 per run) enters this
    function with shape (n_steps, n_runs) and it returns a dict of len(reduce_ops)
    subdicts each with keys named after scalar quantities (loss, accuracy, etc.) holding
    arrays with shape (n_steps,).

    Args:
        events_dict (dict[str, pd.DataFrame]): Dict of arrays to reduce.
        reduce_ops (list[str]): Names of numpy reduce ops. E.g. mean, std, min, max, ...
        verbose (bool, optional): Whether to print progress. Defaults to False.

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

    if verbose:
        print(
            f"Reduced {len(events_dict)} scalars with {len(reduce_ops)} operations:"
            f" ({', '.join(reduce_ops)})"
        )
    return reductions
