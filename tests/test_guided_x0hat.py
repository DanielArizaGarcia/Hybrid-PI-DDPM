import sys
import unittest
from pathlib import Path

import torch
from diffusers import DDPMScheduler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tfm_shells.sampling.guided_x0hat import (
    _predict_original_sample,
    _prepare_clean_x0hat_engineer_query,
)


class GuidedX0HatHelpersTests(unittest.TestCase):
    def test_predict_original_sample_matches_scheduler_for_v_prediction(self) -> None:
        torch.manual_seed(0)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
        )
        sample = torch.randn((2, 1, 4, 4), dtype=torch.float32)
        model_output = torch.randn_like(sample)
        timestep = torch.tensor(321, dtype=torch.long)

        expected = scheduler.step(model_output, timestep, sample).pred_original_sample
        actual = _predict_original_sample(
            scheduler=scheduler,
            sample=sample,
            model_output=model_output,
            timestep=timestep,
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_prepare_clean_x0hat_query_uses_predicted_x0_and_zero_timestep(self) -> None:
        torch.manual_seed(1)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
        )
        sample = torch.randn((3, 1, 4, 4), dtype=torch.float32)
        model_output = torch.randn_like(sample)
        timestep = torch.tensor(500, dtype=torch.long)
        arch_stats = {"z_min": -2.0, "z_max": 3.0}
        eng_stats = {"z_min": 10.0, "z_max": 20.0}

        x0hat_eng, t_batch_eng = _prepare_clean_x0hat_engineer_query(
            scheduler=scheduler,
            sample=sample,
            model_output=model_output,
            timestep=timestep,
            batch_size=sample.shape[0],
            arch_stats=arch_stats,
            eng_stats=eng_stats,
        )

        expected_x0hat_arch = scheduler.step(model_output, timestep, sample).pred_original_sample
        x0hat_real = ((expected_x0hat_arch + 1.0) / 2.0) * (
            arch_stats["z_max"] - arch_stats["z_min"]
        ) + arch_stats["z_min"]
        expected_x0hat_eng = 2.0 * (
            (x0hat_real - eng_stats["z_min"])
            / (eng_stats["z_max"] - eng_stats["z_min"] + 1e-8)
        ) - 1.0

        self.assertTrue(torch.allclose(x0hat_eng, expected_x0hat_eng, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.equal(t_batch_eng, torch.zeros(sample.shape[0], dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
