from __future__ import annotations

import pandas as pd

from tensorboard_reducer import reduce_events


def test_reduce_events(events_dict: dict[str, pd.DataFrame]) -> None:
    reduce_ops = ["mean", "std", "max", "min"]
    reduced_events = reduce_events(events_dict, reduce_ops)

    outkeys = list(reduced_events.keys())
    assert reduce_ops == outkeys, (
        "key mismatch between initial and reduced "
        f"events dict: {reduce_ops=} vs {outkeys=}"
    )

    # loop over reduce operations
    for (op, out_dict), in_arr in zip(reduced_events.items(), events_dict.values()):
        n_steps = len(in_arr)  # length of TB logs

        # loop over event tags (only 'strict/foo' here)
        for tag, out_arr in out_dict.items():
            assert (
                tag in events_dict.keys()
            ), f"unexpected key {tag} in reduced event dict[{op}] = {out_dict.keys()}"

            out_steps = len(out_arr)

            assert (
                n_steps == out_steps
            ), f"length mismatch in initial ({n_steps}) and reduced ({out_steps}) steps"

    min_data = reduced_events["min"]["strict/foo"]
    max_data = reduced_events["max"]["strict/foo"]
    assert all(
        min_data <= max_data
    ), "min reduction was not <= max reduction at every step"
