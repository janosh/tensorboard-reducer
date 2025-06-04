from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tensorboard_reducer import reduce_events

if TYPE_CHECKING:
    from collections.abc import Sequence


def generate_sample_data(
    n_tags: int = 1, n_runs: int = 10, n_steps: int = 5
) -> dict[str, pd.DataFrame]:
    """Generate sample test data for testing reduce operations."""
    events_dict = {}
    rng = np.random.default_rng()
    for idx in range(n_tags):
        data = rng.random((n_steps, n_runs))
        df_rand = pd.DataFrame(data, columns=[f"run_{j}" for j in range(n_runs)])
        events_dict[f"tag_{idx}"] = df_rand
    return events_dict


@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("reduce_ops", [["mean"], ["max", "min"], ["std", "median"]])
def test_reduce_events(
    events_dict: dict[str, pd.DataFrame],
    reduce_ops: Sequence[str],
    verbose: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test reduce_events with sequence of operations."""
    reduced_events = reduce_events(events_dict, reduce_ops, verbose=verbose)

    out_keys = list(reduced_events)
    assert reduce_ops == out_keys, (
        "key mismatch between initial and reduced "
        f"events dict: {reduce_ops=} vs {out_keys=}"
    )

    # loop over reduce operations
    for op in reduce_ops:
        out_dict = reduced_events[op]

        # loop over event tags (e.g., 'strict/foo')
        for tag, out_arr in out_dict.items():
            assert tag in events_dict, (
                f"unexpected key {tag} in reduced event dict[{op}] = {list(out_dict)}"
            )

            in_arr = events_dict[tag]
            n_steps = len(in_arr)  # length of TB logs
            out_steps = len(out_arr)

            assert n_steps == out_steps, (
                f"length mismatch in initial={n_steps} and reduced={out_steps} steps"
            )

    if {"min", "max"} <= set(reduce_ops):
        min_data = reduced_events["min"]["strict/foo"]
        max_data = reduced_events["max"]["strict/foo"]
        assert all(min_data <= max_data), (
            "min reduction was not <= max reduction at every step"
        )

    stdout, stderr = capsys.readouterr()
    assert stderr == ""
    if verbose:
        assert (
            f"Reduced {len(events_dict)} scalars with {len(reduce_ops)} operations:"
            f" ({', '.join(reduce_ops)})"
        ) in stdout
    else:
        assert stdout == ""


@pytest.mark.parametrize("reduce_op", ["mean", "std", "max"])
def test_reduce_events_reduce_op_str_list_equivalence(reduce_op: str) -> None:
    """Test string input (fixes issue #44) and equivalence with list input."""
    events_dict = generate_sample_data(n_tags=2, n_runs=3, n_steps=4)

    # Test string input
    result_string = reduce_events(events_dict, reduce_op)

    # Test list input
    result_list = reduce_events(events_dict, [reduce_op])

    # Verify string input structure and correctness
    assert len(result_string) == 1
    assert reduce_op in result_string
    for tag, df in events_dict.items():
        assert tag in result_string[reduce_op]
        expected = getattr(df, reduce_op)(axis=1)
        pd.testing.assert_series_equal(result_string[reduce_op][tag], expected)

    # Verify string and list inputs produce identical results
    assert result_string.keys() == result_list.keys()
    for tag in events_dict:
        pd.testing.assert_series_equal(
            result_string[reduce_op][tag], result_list[reduce_op][tag]
        )


@pytest.mark.parametrize("n_tags, n_runs, n_steps", [(1, 10, 5), (2, 5, 3), (3, 3, 10)])
def test_reduce_events_dimensions(n_tags: int, n_runs: int, n_steps: int) -> None:
    """Test reduce_events with different data dimensions."""
    events_dict = generate_sample_data(n_tags=n_tags, n_runs=n_runs, n_steps=n_steps)
    reduce_ops = ["mean", "std", "max", "min"]
    reduced_events = reduce_events(events_dict, reduce_ops)

    for op, reduced_dict in reduced_events.items():
        assert len(reduced_dict) == n_tags, (
            f"mismatch in number of tags for operation {op}: "
            f"expected {n_tags}, found {len(reduced_dict)}"
        )

        for tag, reduced_df in reduced_dict.items():
            assert len(reduced_df) == n_steps, (
                f"length mismatch in initial={n_steps} vs reduced={len(reduced_df)}"
                f" steps for tag {tag} and operation {op}"
            )


@pytest.mark.parametrize("reduce_ops", [["mean"], ["max", "min"], ["std", "median"]])
def test_reduce_events_empty_input(reduce_ops: Sequence[str]) -> None:
    """Test reduce_events with empty input dictionary."""
    reduced_events = reduce_events({}, reduce_ops)
    assert reduced_events == {op: {} for op in reduce_ops}
