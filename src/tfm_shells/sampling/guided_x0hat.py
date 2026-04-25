from __future__ import annotations

from pathlib import Path
from typing import Any

from tfm_shells.utils.matplotlib_backend import configure_matplotlib_backend

configure_matplotlib_backend()

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from tfm_shells.config import load_config, resolve_project_path, save_config
from tfm_shells.models.factory import build_scheduler, build_unet
from tfm_shells.sampling.guided import (
    _bell_or_poly,
    _load_checkpoint,
    _normalize_minmax,
    _renormalize_tensor,
)
from tfm_shells.training.common import (
    format_metric,
    make_run_name,
    prepare_run_directories,
    resolve_device,
    seed_everything,
)
from tfm_shells.utils.io import save_json
from tfm_shells.utils.physics import compute_membrane_factor_map_from_real_physics
from tfm_shells.utils.tracking import ExperimentTracker


def _expand_to_sample_dims(values: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    if values.ndim == 0:
        values = values.view(1)
    return values.reshape(-1, *([1] * (sample.ndim - 1))).to(device=sample.device, dtype=sample.dtype)


def _predict_original_sample(
    scheduler: Any,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
) -> torch.Tensor:
    prediction_type = str(scheduler.config.prediction_type)
    alpha_prod_t = _expand_to_sample_dims(scheduler.alphas_cumprod[timestep], sample)
    beta_prod_t = 1.0 - alpha_prod_t

    if prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif prediction_type == "sample":
        pred_original_sample = model_output
    elif prediction_type == "v_prediction":
        pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")

    if bool(getattr(scheduler.config, "thresholding", False)):
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif bool(getattr(scheduler.config, "clip_sample", True)):
        clip_range = float(getattr(scheduler.config, "clip_sample_range", 1.0))
        pred_original_sample = pred_original_sample.clamp(-clip_range, clip_range)

    return pred_original_sample


def _prepare_clean_x0hat_engineer_query(
    scheduler: Any,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
    batch_size: int,
    arch_stats: dict[str, Any],
    eng_stats: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    x0hat_arch = _predict_original_sample(
        scheduler=scheduler,
        sample=sample,
        model_output=model_output,
        timestep=timestep,
    )
    x0hat_eng = _renormalize_tensor(
        x0hat_arch,
        src_min=float(arch_stats["z_min"]),
        src_max=float(arch_stats["z_max"]),
        dst_min=float(eng_stats["z_min"]),
        dst_max=float(eng_stats["z_max"]),
    )
    t_batch_eng = torch.zeros(batch_size, device=sample.device, dtype=torch.long)
    return x0hat_eng, t_batch_eng


def run_guided_sampling_x0hat(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    directories = prepare_run_directories(config, role="sample")
    runtime_cfg = config.get("runtime", {})
    device = resolve_device(str(runtime_cfg.get("device", "auto")))
    save_config(config, directories["run_root"] / "config.yaml")
    save_config(config, directories["model_root"] / "config.yaml")

    architect_ckpt = _load_checkpoint(resolve_project_path(config, config["architect"]["checkpoint"]), device)
    engineer_ckpt = _load_checkpoint(resolve_project_path(config, config["engineer"]["checkpoint"]), device)

    architect = build_unet(architect_ckpt["model_config"]).to(device)
    architect.load_state_dict(architect_ckpt["model_state_dict"])
    architect.eval()

    engineer = build_unet(engineer_ckpt["model_config"]).to(device)
    engineer.load_state_dict(engineer_ckpt["model_state_dict"])
    engineer.eval()

    scheduler = build_scheduler(architect_ckpt["model_config"])
    scheduler.set_timesteps(int(config["sampling"]["num_inference_steps"]), device=device)

    source_file = resolve_project_path(config, config["conditioning"]["source_file"])
    with np.load(source_file) as data:
        fz_real = data["fz"].astype(np.float32)

    batch_size = int(config["conditioning"]["batch_size"])
    arch_stats = architect_ckpt["normalization_stats"]
    eng_stats = engineer_ckpt["normalization_stats"]
    fz_norm_eng = _normalize_minmax(fz_real, float(eng_stats["fz_min"]), float(eng_stats["fz_max"]))
    fz_cond_eng = torch.from_numpy(fz_norm_eng).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)

    x = torch.randn((batch_size, 1, 64, 64), device=device)
    history = {"t": [], "objective": [], "mf_mean": [], "grad_norm": [], "guide_weight": []}

    p_mean = torch.tensor(eng_stats["physics_mean"], dtype=torch.float32, device=device).unsqueeze(0)
    p_std = torch.tensor(eng_stats["physics_std"], dtype=torch.float32, device=device).unsqueeze(0)

    total_steps = len(scheduler.timesteps)
    progress = tqdm(scheduler.timesteps, desc=f"sample 0000/{total_steps:04d}", leave=True)
    for index, timestep in enumerate(progress, start=1):
        t_value = int(timestep.item())
        t_batch = torch.full((x.shape[0],), t_value, device=device, dtype=torch.long)

        with torch.no_grad():
            model_output = architect(x, t_batch).sample

        x_req = x.detach().clone().requires_grad_(True)
        model_output_for_guidance = architect(x_req, t_batch).sample
        x0hat_eng, t_batch_eng = _prepare_clean_x0hat_engineer_query(
            scheduler=scheduler,
            sample=x_req,
            model_output=model_output_for_guidance,
            timestep=timestep,
            batch_size=batch_size,
            arch_stats=arch_stats,
            eng_stats=eng_stats,
        )
        engineer_input = (
            torch.cat([x0hat_eng, fz_cond_eng], dim=1)
            if int(engineer_ckpt["model_config"]["in_channels"]) > 1
            else x0hat_eng
        )
        pred_phys_norm = engineer(engineer_input, t_batch_eng).sample
        pred_phys_real = pred_phys_norm * p_std + p_mean
        mf_map = compute_membrane_factor_map_from_real_physics(pred_phys_real)
        mf_mean = mf_map.mean(dim=(1, 2, 3))
        objective_per_sample = (1.0 - mf_mean) ** 2
        objective_for_grad = objective_per_sample.sum()
        objective = objective_per_sample.mean()

        grad = torch.autograd.grad(objective_for_grad, x_req, retain_graph=False, create_graph=False)[0]
        guide_weight = _bell_or_poly(config, index - 1, total_steps)
        grad = torch.clamp(
            guide_weight * float(config["sampling"]["guidance_scale"]) * grad,
            min=-float(config["sampling"]["grad_clip"]),
            max=float(config["sampling"]["grad_clip"]),
        )
        grad_norm = float(grad.flatten(1).norm(dim=1).mean().item())
        mf_mean_value = float(mf_mean.mean().item())

        alpha_bar_t = scheduler.alphas_cumprod[timestep].to(device)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
        guided_output = model_output + sqrt_one_minus_alpha_bar * grad
        x = scheduler.step(guided_output, timestep, x).prev_sample

        progress.set_description(f"sample {index:04d}/{total_steps:04d}")
        progress.set_postfix(
            grad_norm=format_metric(grad_norm),
            w=format_metric(float(guide_weight)),
            mf=format_metric(mf_mean_value),
            refresh=False,
        )
        history["t"].append(t_value)
        history["objective"].append(float(objective.item()))
        history["mf_mean"].append(mf_mean_value)
        history["grad_norm"].append(grad_norm)
        history["guide_weight"].append(float(guide_weight))

    z_min = float(arch_stats["z_min"])
    z_max = float(arch_stats["z_max"])
    samples_real = ((x.detach().cpu().numpy() + 1.0) / 2.0) * (z_max - z_min) + z_min

    output_npz = directories["run_root"] / "guided_samples.npz"
    np.savez_compressed(output_npz, z=samples_real)

    figure, axes = plt.subplots(2, int(np.ceil(samples_real.shape[0] / 2.0)), figsize=(16, 6))
    axes = np.array(axes).reshape(-1)
    for idx, axis in enumerate(axes):
        axis.axis("off")
        if idx < samples_real.shape[0]:
            axis.imshow(samples_real[idx, 0], cmap="viridis")
            axis.set_title(f"sample_{idx}")
    figure.suptitle("Guided conditional samples (clean x0-hat baseline)")
    figure.tight_layout()
    samples_png = directories["run_root"] / "guided_samples.png"
    figure.savefig(samples_png, dpi=180, bbox_inches="tight")
    plt.close(figure)

    history_png = directories["run_root"] / "guided_history.png"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(history["t"], history["mf_mean"])
    ax[0].invert_xaxis()
    ax[0].set_title("mf_mean")
    ax[0].grid(alpha=0.3)
    ax[1].plot(history["t"], history["objective"])
    ax[1].invert_xaxis()
    ax[1].set_title("objective")
    ax[1].grid(alpha=0.3)
    ax[2].plot(history["t"], history["grad_norm"])
    ax[2].invert_xaxis()
    ax[2].set_title("grad_norm")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(history_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    run_name = make_run_name(config, role="sample")
    with ExperimentTracker(config, directories["project_root"], run_name) as tracker:
        tracker.log_config(config)
        tracker.log_artifact(directories["run_root"] / "config.yaml", artifact_path="run")
        tracker.log_metrics(
            {
                "samples_generated": float(samples_real.shape[0]),
                "final_objective": float(history["objective"][-1]),
                "final_mf_mean": float(history["mf_mean"][-1]),
                "final_grad_norm": float(history["grad_norm"][-1]),
            }
        )
        tracker.log_artifact(output_npz, artifact_path="samples")
        tracker.log_artifact(samples_png, artifact_path="samples")
        tracker.log_artifact(history_png, artifact_path="samples")

    summary = {
        "config_file": str(directories["run_root"] / "config.yaml"),
        "samples_file": str(output_npz),
        "plot_file": str(samples_png),
        "history_plot_file": str(history_png),
        "guidance_query_mode": "pred_original_sample_clean_engineer",
        "engineer_query_timestep": 0,
        "samples_generated": int(samples_real.shape[0]),
        "sample_shape": list(samples_real.shape),
        "conditioning_source_file": str(source_file),
        "num_inference_steps": int(config["sampling"]["num_inference_steps"]),
        "final_mf_mean": float(history["mf_mean"][-1]),
    }
    save_json(summary, directories["run_root"] / "summary.json")
    save_json(summary, directories["model_root"] / "summary.json")
    return summary
