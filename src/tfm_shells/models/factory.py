from __future__ import annotations

from typing import Any

import torch
from diffusers import DDPMScheduler, UNet2DModel

from tfm_shells.models.equino import EquiNOModel
from tfm_shells.models.parallel_pb_unet import ParallelPBUNet
from tfm_shells.models.shell_weakrefine_operator import ShellWeakRefineOperator
from tfm_shells.models.ufno import UFNOModel
from tfm_shells.models.transformer_operator import TransformerOperator


def build_unet(model_config: dict[str, Any]) -> torch.nn.Module:
    kind = str(model_config.get("kind", "unet"))
    if kind == "parallel_pb_unet":
        return ParallelPBUNet(
            sample_size=int(model_config["sample_size"]),
            in_channels=int(model_config["in_channels"]),
            layers_per_block=int(model_config["layers_per_block"]),
            block_out_channels=tuple(int(value) for value in model_config["block_out_channels"]),
            down_block_types=tuple(model_config["down_block_types"]),
            up_block_types=tuple(model_config["up_block_types"]),
            branch_channels=model_config.get("branch_channels"),
        )
    if kind == "equino":
        return EquiNOModel(
            sample_size=int(model_config["sample_size"]),
            in_channels=int(model_config["in_channels"]),
            out_channels=int(model_config["out_channels"]),
            operator_width=int(model_config.get("operator_width", 128)),
            num_operator_layers=int(model_config.get("num_operator_layers", 6)),
            spectral_modes_height=int(model_config.get("spectral_modes_height", 16)),
            spectral_modes_width=int(model_config.get("spectral_modes_width", 16)),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 256)),
            head_hidden_channels=int(model_config.get("head_hidden_channels", 128)),
            modal_rank=int(model_config.get("modal_rank", 12)),
            modal_residual_weight=float(model_config.get("modal_residual_weight", 0.25)),
            branch_channels=model_config.get("branch_channels"),
            use_coordinate_grid=bool(model_config.get("use_coordinate_grid", True)),
            predict_log_variance=bool(model_config.get("predict_log_variance", False)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
    if kind == "ufno":
        return UFNOModel(
            sample_size=int(model_config["sample_size"]),
            in_channels=int(model_config["in_channels"]),
            out_channels=int(model_config["out_channels"]),
            operator_width=int(model_config.get("operator_width", 128)),
            num_operator_layers=int(model_config.get("num_operator_layers", 6)),
            spectral_modes_height=int(model_config.get("spectral_modes_height", 16)),
            spectral_modes_width=int(model_config.get("spectral_modes_width", 16)),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 256)),
            head_hidden_channels=int(model_config.get("head_hidden_channels", 128)),
            local_kernel=int(model_config.get("local_kernel", 3)),
            use_unet_branch=bool(model_config.get("use_unet_branch", True)),
            branch_channels=model_config.get("branch_channels"),
            use_coordinate_grid=bool(model_config.get("use_coordinate_grid", True)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
    if kind == "dit_operator":
        return TransformerOperator(
            sample_size=int(model_config["sample_size"]),
            in_channels=int(model_config["in_channels"]),
            out_channels=int(model_config["out_channels"]),
            patch_size=int(model_config.get("patch_size", 4)),
            embed_dim=int(model_config.get("embed_dim", 640)),
            depth=int(model_config.get("depth", 12)),
            num_heads=int(model_config.get("num_heads", 10)),
            mlp_ratio=float(model_config.get("mlp_ratio", 4.0)),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 256)),
            use_coordinate_grid=bool(model_config.get("use_coordinate_grid", True)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
    if kind == "shell_weakrefine_operator":
        return ShellWeakRefineOperator(
            sample_size=int(model_config["sample_size"]),
            in_channels=int(model_config["in_channels"]),
            out_channels=int(model_config["out_channels"]),
            operator_width=int(model_config.get("operator_width", 128)),
            num_operator_layers=int(model_config.get("num_operator_layers", 6)),
            spectral_modes_height=int(model_config.get("spectral_modes_height", 16)),
            spectral_modes_width=int(model_config.get("spectral_modes_width", 16)),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 256)),
            branch_hidden_channels=int(model_config.get("branch_hidden_channels", 128)),
            branch_channels=model_config.get("branch_channels"),
            use_coordinate_grid=bool(model_config.get("use_coordinate_grid", True)),
            predict_log_variance=bool(model_config.get("predict_log_variance", True)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
    return UNet2DModel(
        sample_size=int(model_config["sample_size"]),
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        layers_per_block=int(model_config["layers_per_block"]),
        block_out_channels=tuple(int(value) for value in model_config["block_out_channels"]),
        down_block_types=tuple(model_config["down_block_types"]),
        up_block_types=tuple(model_config["up_block_types"]),
    )


def build_scheduler(model_config: dict[str, Any]) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule=str(model_config["beta_schedule"]),
        prediction_type=str(model_config["prediction_type"]),
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
