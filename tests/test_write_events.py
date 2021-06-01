from os.path import isdir
from shutil import rmtree

from tensorboard_reducer import reduce_events, write_tb_events


reduce_ops = ["mean", "std"]


def test_write_tb_events(events_dict):
    rmtree("tmp", ignore_errors=True)

    reduced_events = reduce_events(events_dict, reduce_ops)

    write_tb_events(reduced_events, "tmp/reduced")

    for op in reduce_ops:
        assert isdir(f"tmp/reduced-{op}"), f"couldn't find {op} reduction outdir"

    write_tb_events(reduced_events, "tmp/reduced", overwrite=True)

    # will clean up or raise FileNotFoundError if directory unexpectedly does not exist
    rmtree("tmp/reduced-mean")
    rmtree("tmp/reduced-std")
