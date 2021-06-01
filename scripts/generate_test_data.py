import os

import torch
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(0)


def gen_strict_test_data() -> None:
    """Generate random event files for testing TensorBoard Reducer in strict mode,
    i.e. with different runs containing identical tags and equal number of steps.
    """
    os.makedirs("tests/runs/strict", exist_ok=True)

    for idx in range(1, 4):
        writer = SummaryWriter(f"tests/runs/strict/run_{idx}")

        n_steps = 100

        for step in range(n_steps):
            writer.add_scalar("strict/foo", torch.rand([]) + idx, 5 * step)

        writer.close()


def gen_lax_test_data() -> None:
    """Generate random event files for testing TensorBoard Reducer in lax mode,
    i.e. with different runs containing different tags and unequal number of steps.
    """
    os.makedirs("tests/runs/lax", exist_ok=True)

    for idx in range(1, 4):
        writer = SummaryWriter(f"tests/runs/lax/run_{idx}")

        n_steps = 100 + idx * 10

        for step in range(n_steps):
            writer.add_scalar("lax/foo", torch.rand([]) + idx, step)

            writer.add_scalar(f"lax/bar_{idx}", torch.rand([]) + idx, step)

            writer.add_scalar(f"lax/bar_{idx + 1}", torch.rand([]) + idx, step)

        writer.close()


if __name__ == "__main__":
    gen_strict_test_data()
    gen_lax_test_data()
