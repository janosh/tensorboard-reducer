import numpy as np

from tensorboard_reducer import reduce_events


def test_reduce_events(events_dict):

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

        # loop over event tags (only 'train/loss' here)
        for tag, out_arr in out_dict.items():

            assert (
                tag in events_dict.keys()
            ), f"unexpected key {tag} in reduced event dict[{op}] = {out_dict.keys()}"

            out_steps = len(out_arr)

            assert (
                n_steps == out_steps
            ), f"len mismatch in initial ({n_steps}) and reduced ({out_steps}) time steps"

            assert np.allclose(fixture_event_dict[op][tag], out_arr, atol=1e-3)

    min_data = reduced_events["min"]["train/loss"]
    max_data = reduced_events["max"]["train/loss"]
    assert all(
        min_data <= max_data
    ), "min reduction was not <= max reduction at every step"


# fmt: off
fixture_event_dict = {
    "mean": {
        "train/loss": np.array([152.1530, 128.0775, 118.0522, 115.4030, 106.6729, 98.1540, 92.8584, 71.7001, 62.8971, 62.2401, 53.1173, 49.8189, 44.8458, 37.4322, 37.7564, 35.2916, 28.8625, 26.4018, 23.1969, 20.1772, 17.9658, 18.0196, 16.4379, 15.5518, 13.2141, 11.3726, 11.2012, 8.8046, 9.2694, 9.2751, 7.5211, 7.1538, 5.8784, 5.4574, 5.2583, 4.1255, 4.3862, 3.7987, 3.3627, 2.6442, 2.5392, 2.5403, 2.2232, 2.0292, 1.8836, 1.7155, 1.5354, 1.3560, 1.2753, 1.0948])  # noqa: E501

    },
    "std": {
        "train/loss": np.array([24.3834, 12.4175, 8.4125, 4.5599, 8.0898, 5.2784, 2.7101, 0.9511, 0.6539, 4.6668, 1.5058, 3.7522, 4.7185, 4.1376, 3.0874, 4.0749, 2.4044, 1.8863, 2.5475, 2.0941, 1.4208, 1.6276, 0.3663, 0.9320, 0.6191, 1.1864, 0.5560, 0.7320, 0.6131, 0.6315, 0.8250, 0.5978, 0.2254, 0.2786, 0.1954, 0.2683, 0.3279, 0.2747, 0.0825, 0.0429, 0.1806, 0.1365, 0.1714, 0.0250, 0.2365, 0.1989, 0.1558, 0.0482, 0.0388, 0.0468])  # noqa: E501
    },
    "max": {
        "train/loss": np.array([180.4095, 145.5871, 129.6668, 121.8313, 118.0129, 101.9177, 96.3889, 73.0449, 63.7794, 68.8201, 55.0784, 54.2755, 50.7787, 43.0440, 40.4129, 39.5267, 32.0034, 28.7553, 26.7042, 22.2032, 19.4587, 20.2811, 16.8584, 16.5316, 14.0886, 12.3369, 11.7545, 9.6722, 10.1162, 10.1682, 8.5254, 7.8689, 6.0565, 5.6761, 5.4770, 4.4287, 4.7561, 4.1730, 3.4269, 2.7009, 2.7937, 2.6813, 2.4183, 2.0609, 2.2179, 1.9937, 1.7557, 1.4013, 1.3207, 1.1412])  # noqa: E501
    },
    "min": {
        "train/loss": np.array([120.9074, 118.1589, 110.0129, 111.7444, 99.6904, 90.6892, 89.8013, 70.9978, 62.2159, 58.5066, 51.4176, 45.0959, 39.2339, 33.1906, 33.4272, 29.7895, 26.1637, 24.1373, 20.7300, 17.2934, 16.0545, 16.5172, 15.9656, 14.2983, 12.7388, 9.7013, 10.4406, 7.8816, 8.6844, 8.8181, 6.5045, 6.4057, 5.5604, 5.0642, 5.0027, 3.7762, 3.9590, 3.5214, 3.2462, 2.5969, 2.3930, 2.3556, 2.0010, 1.9996, 1.7061, 1.5398, 1.4193, 1.2892, 1.2259, 1.0306])  # noqa: E501
    },
}
# fmt: on
