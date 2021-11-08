import os
import sys
from shutil import rmtree

import torch
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)


def gen_strict_test_data() -> None:
    """Generate random event files for testing TensorBoard Reducer in strict mode,
    i.e. with different runs containing identical tags and equal number of steps.
    """
    base_dir = "tests/runs/strict"
    rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    n_steps = 100

    for idx in range(1, 4):
        writer = SummaryWriter(f"{base_dir}/run_{idx}")

        for step in range(n_steps):
            writer.add_scalar("strict/foo", torch.rand([]) + idx, 5 * step)

        writer.close()


def gen_lax_test_data() -> None:
    """Generate random event files for testing TensorBoard Reducer in lax mode,
    i.e. with different runs containing different tags and unequal number of steps.
    """
    base_dir = "tests/runs/lax"
    rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    for idx in range(1, 4):
        writer = SummaryWriter(f"{base_dir}/run_{idx}")

        n_steps = 100 + idx * 10

        for step in range(n_steps):
            writer.add_scalar("lax/foo", torch.rand([]) + idx, step)

            writer.add_scalar(f"lax/bar_{idx}", torch.rand([]) + idx, step)

            writer.add_scalar(f"lax/bar_{idx + 1}", torch.rand([]) + idx, step)

        writer.close()


def gen_dup_steps_test_data() -> None:
    """Generate random event files for testing TensorBoard Reducer in handle-duplicate-steps
    mode, i.e. with a runs containing multiple values for a single tag at the same step.
    """
    base_dir = "tests/runs/duplicate_steps"
    rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    n_steps = 100

    for idx in range(1, 4):
        writer = SummaryWriter(f"{base_dir}/run_{idx}")

        # write value at every step
        for step in range(n_steps):
            writer.add_scalar("dup_steps/foo", torch.rand([]) + idx, step)

            writer.add_scalar("dup_steps/bar", torch.rand([]) + idx, step)

        # create a duplicate at every 10th step
        for step in range(n_steps - 10, 2 * n_steps):
            writer.add_scalar("dup_steps/foo", torch.rand([]) + idx, step)

            writer.add_scalar("dup_steps/bar", torch.rand([]) + idx, step)

        writer.close()


if __name__ == "__main__":
    # call as
    # python scripts/generate_test_data.py strict|lax|...
    if sys.argv[1] == "strict":
        print("Generating strict TB runs: same tags and step counts for different runs")
        gen_strict_test_data()
    elif sys.argv[1] == "lax":
        print(
            "Generating lax TB runs: different tags and step counts for different runs"
        )
        gen_lax_test_data()
    elif sys.argv[1] == "dup_steps":
        print("Generating TB runs with duplicate steps")
        gen_dup_steps_test_data()
