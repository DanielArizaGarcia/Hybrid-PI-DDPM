import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tfm_shells.training.train_engineer import _sample_engineer_noisy_inputs


class _FakeScheduler:
    def __init__(self) -> None:
        self.calls = 0

    def add_noise(self, z_clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return z_clean + 0.5 * noise


class SampleEngineerNoisyInputsTests(unittest.TestCase):
    def test_clean_only_uses_zero_timesteps_and_skips_scheduler_noise(self) -> None:
        z_clean = torch.ones((3, 1, 4, 4), dtype=torch.float32)
        scheduler = _FakeScheduler()

        timesteps, z_noisy = _sample_engineer_noisy_inputs(
            z_clean=z_clean,
            scheduler=scheduler,
            t_max=1000,
            device=torch.device("cpu"),
            noise_mode="clean_only",
        )

        self.assertTrue(torch.equal(timesteps, torch.zeros(3, dtype=torch.long)))
        self.assertTrue(torch.equal(z_noisy, z_clean))
        self.assertEqual(scheduler.calls, 0)

    def test_diffused_samples_timesteps_and_calls_scheduler(self) -> None:
        torch.manual_seed(0)
        z_clean = torch.ones((3, 1, 4, 4), dtype=torch.float32)
        scheduler = _FakeScheduler()

        timesteps, z_noisy = _sample_engineer_noisy_inputs(
            z_clean=z_clean,
            scheduler=scheduler,
            t_max=1000,
            device=torch.device("cpu"),
            noise_mode="diffused",
        )

        self.assertEqual(timesteps.shape, torch.Size([3]))
        self.assertEqual(timesteps.dtype, torch.long)
        self.assertTrue(bool(torch.all(timesteps >= 0)))
        self.assertTrue(bool(torch.all(timesteps < 1000)))
        self.assertEqual(scheduler.calls, 1)
        self.assertFalse(torch.equal(z_noisy, z_clean))


if __name__ == "__main__":
    unittest.main()
