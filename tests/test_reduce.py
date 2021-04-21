from tensorboard_reducer import load_tb_events, reduce_events


def test_read_events():
    events_dict = load_tb_events("tests/tensorboard_runs/run_*")

    reduce_ops = ["mean", "std", "max", "min"]
    reduced_events = reduce_events(events_dict, reduce_ops)

    inkeys, outkeys = events_dict.keys(), list(reduced_events.keys())
    assert (
        reduce_ops == outkeys
    ), f"key mismatch between initial and reduced events dict: {reduce_ops=}, {outkeys=}"

    for (op, in_arr), out_dict in zip(events_dict.items(), reduced_events.values()):

        n_timesteps, _ = in_arr.shape

        for tag, out_arr in out_dict.items():
            assert (
                tag in inkeys
            ), f"unexpected key {tag} in reduced event dict[{op}] = {out_dict.keys()}"
            (out_steps,) = out_arr.shape
            assert (
                n_timesteps == out_steps
            ), f"len mismatch in initial ({n_timesteps}) and reduced ({out_steps}) time steps"
