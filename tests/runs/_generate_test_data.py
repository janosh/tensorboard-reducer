import os
from shutil import rmtree

import torch
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)

ROOT = os.path.dirname(__file__)


def generate_strict_test_data() -> None:
    """Generate strict TB runs: same tags and step counts for different runs

    Generate random event files for testing TensorBoard Reducer in strict mode,
    i.e. with different runs containing identical tags and equal number of steps.
    """
    out_dir = f"{ROOT}/strict"
    rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    n_steps = 100

    for idx in range(1, 4):
        writer = SummaryWriter(f"{out_dir}/run_{idx}")

        for step in range(n_steps):
            writer.add_scalar("strict/foo", torch.rand([]) + idx, 5 * step)

        writer.close()


def generate_lax_test_data() -> None:
    """Generate lax TB runs: different tags and step counts for different runs

    Writes random event files for testing TensorBoard Reducer in --lax-steps and/or
    --lax-tags mode, i.e. with different runs containing different tags and unequal
    number of steps.
    """
    out_dir = f"{ROOT}/lax"
    rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(1, 4):
        writer = SummaryWriter(f"{out_dir}/run_{idx}")

        n_steps = 100 + idx * 10

        for step in range(n_steps):
            writer.add_scalar("lax/foo", torch.rand([]) + idx, step)

            writer.add_scalar(f"lax/bar_{idx}", torch.rand([]) + idx, step)

            writer.add_scalar(f"lax/bar_{idx + 1}", torch.rand([]) + idx, step)

        writer.close()


def generate_duplicate_steps_test_data() -> None:
    """Generate TB runs with duplicate steps

    Writes random TensorBoard event files for testing tb-reducer in --handle-dup-steps
    mode, i.e. with a runs containing multiple values for a single tag at the same step.
    """
    out_dir = f"{ROOT}/duplicate_steps"
    rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    n_steps = 100

    for idx in range(1, 4):
        writer = SummaryWriter(f"{out_dir}/run_{idx}")

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
    # call as `python scripts/generate_test_data.py`
    generate_strict_test_data()
    generate_lax_test_data()
    generate_duplicate_steps_test_data()
