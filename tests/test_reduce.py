from __future__ import annotations

import pandas as pd
import pytest

from tensorboard_reducer import reduce_events


@pytest.mark.parametrize("verbose", [True, False])
def test_reduce_events(
    events_dict: dict[str, pd.DataFrame],
    verbose: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reduce_ops = ["mean", "std", "max", "min"]
    reduced_events = reduce_events(events_dict, reduce_ops, verbose=verbose)

    out_keys = list(reduced_events)
    assert reduce_ops == out_keys, (
        "key mismatch between initial and reduced "
        f"events dict: {reduce_ops=} vs {out_keys=}"
    )

    # loop over reduce operations
    for (op, out_dict), in_arr in zip(reduced_events.items(), events_dict.values()):
        n_steps = len(in_arr)  # length of TB logs

        # loop over event tags (only 'strict/foo' here)
        for tag, out_arr in out_dict.items():
            assert (
                tag in events_dict
            ), f"unexpected key {tag} in reduced event dict[{op}] = {list(out_dict)}"

            out_steps = len(out_arr)

            assert (
                n_steps == out_steps
            ), f"length mismatch in initial ({n_steps}) and reduced ({out_steps}) steps"

    min_data = reduced_events["min"]["strict/foo"]
    max_data = reduced_events["max"]["strict/foo"]
    assert all(
        min_data <= max_data
    ), "min reduction was not <= max reduction at every step"

    stdout, stderr = capsys.readouterr()
    assert stderr == ""
    if verbose:
        assert (
            f"Reduced {len(events_dict)} scalars with {len(reduce_ops)} operations:"
            f" ({', '.join(reduce_ops)})"
        ) in stdout
    else:
        assert stdout == ""
