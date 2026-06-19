from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tfm_shells.constants import PHYSICS_KEYS


def _normalize_minmax(array: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    scale = maximum - minimum
    if abs(scale) < 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (2.0 * ((array - minimum) / scale) - 1.0).astype(np.float32)


def _physics_stack(data: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.concatenate([data[key].astype(np.float32) for key in PHYSICS_KEYS], axis=0)


# --- Augmentacion por simetria diedrica D4 (validada en scripts/validate_d4_symmetry.py) ---
# El dataset (sin agujero, carga fz uniforme) es D4-simetrico. Cada elemento de D4 = transformacion
# espacial (rot/flip sobre los dos ultimos ejes) + accion sobre las componentes de los tensores de
# rango 2 (T11, T22, T12) que preserva la contraccion fisica T:S y, por tanto, el factor de membrana.
# Bloques tensoriales dentro de los 13 canales fisicos: se(1,2,3) sf(4,5,6) sk(7,8,9) sm(10,11,12);
# el canal 0 (uz) es escalar -> solo transformacion espacial.
_TENSOR_BLOCKS = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))

# (nombre, op_espacial, swap_ab, signo_c)
_D4 = (
    ("identity", "id", False, 1.0),
    ("rot90", "rot90", True, -1.0),
    ("rot180", "rot180", False, 1.0),
    ("rot270", "rot270", True, -1.0),
    ("fliplr", "fliplr", False, -1.0),
    ("flipud", "flipud", False, -1.0),
    ("transpose", "transpose", True, 1.0),
    ("antidiag", "antidiag", True, 1.0),
)


def _sp_apply(array: np.ndarray, op: str) -> np.ndarray:
    """Transformacion espacial de D4 sobre los dos ultimos ejes (canales delante)."""
    if op == "id":
        return array
    if op == "rot90":
        return np.rot90(array, 1, axes=(-2, -1))
    if op == "rot180":
        return np.rot90(array, 2, axes=(-2, -1))
    if op == "rot270":
        return np.rot90(array, 3, axes=(-2, -1))
    if op == "fliplr":
        return np.flip(array, axis=-1)
    if op == "flipud":
        return np.flip(array, axis=-2)
    if op == "transpose":
        return np.swapaxes(array, -1, -2)
    if op == "antidiag":
        return np.rot90(np.flip(array, axis=-1), 1, axes=(-2, -1))
    raise ValueError(f"D4 op desconocida: {op}")


def _apply_tensor_components(physics: np.ndarray, swap_ab: bool, sign_c: float) -> np.ndarray:
    """Permuta/escala las componentes (T11, T22, T12) de cada bloque tensorial. Devuelve copia."""
    out = physics.copy()
    for i11, i22, i12 in _TENSOR_BLOCKS:
        if swap_ab:
            out[i11] = physics[i22]
            out[i22] = physics[i11]
        out[i12] = sign_c * physics[i12]
    return out


def _random_d4() -> tuple[str, bool, float]:
    # torch.randint queda correctamente re-sembrado por DataLoader en cada worker (numpy no).
    idx = int(torch.randint(0, len(_D4), (1,)).item())
    _, op, swap_ab, sign_c = _D4[idx]
    return op, swap_ab, sign_c


def compute_normalization_stats(records: list[dict[str, Any]], include_physics: bool) -> dict[str, Any]:
    z_mins: list[float] = []
    z_maxs: list[float] = []
    fz_mins: list[float] = []
    fz_maxs: list[float] = []

    p_sum = np.zeros(len(PHYSICS_KEYS), dtype=np.float64)
    p_sq_sum = np.zeros(len(PHYSICS_KEYS), dtype=np.float64)
    pixel_count = 0

    for record in records:
        with np.load(record["path"]) as data:
            z = data["z"].astype(np.float64)
            fz = data["fz"].astype(np.float64)
            z_mins.append(float(z.min()))
            z_maxs.append(float(z.max()))
            fz_mins.append(float(fz.min()))
            fz_maxs.append(float(fz.max()))

            if include_physics:
                for index, key in enumerate(PHYSICS_KEYS):
                    arr = data[key].astype(np.float64)
                    p_sum[index] += arr.sum()
                    p_sq_sum[index] += np.square(arr).sum()
                pixel_count += int(np.prod(data["z"].shape[1:]))

    stats: dict[str, Any] = {
        "z_min": min(z_mins),
        "z_max": max(z_maxs),
        "fz_min": min(fz_mins),
        "fz_max": max(fz_maxs),
    }

    if include_physics:
        mean = (p_sum / pixel_count).reshape(len(PHYSICS_KEYS), 1, 1)
        variance = np.maximum((p_sq_sum / pixel_count) - np.square(p_sum / pixel_count), 1e-12)
        std = np.sqrt(variance).reshape(len(PHYSICS_KEYS), 1, 1)
        stats["physics_mean"] = mean.astype(np.float32).tolist()
        stats["physics_std"] = std.astype(np.float32).tolist()
    return stats


class ShellDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        stats: dict[str, Any],
        include_physics: bool,
        augment_d4: bool = False,
    ) -> None:
        self.records = records
        self.stats = stats
        self.include_physics = include_physics
        self.augment_d4 = bool(augment_d4)
        self.z_min = float(stats["z_min"])
        self.z_max = float(stats["z_max"])
        self.fz_min = float(stats["fz_min"])
        self.fz_max = float(stats["fz_max"])
        self.physics_mean = None
        self.physics_std = None
        if include_physics:
            self.physics_mean = np.asarray(stats["physics_mean"], dtype=np.float32)
            self.physics_std = np.asarray(stats["physics_std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        path = Path(record["path"])
        with np.load(path) as data:
            z = data["z"].astype(np.float32)
            fz = data["fz"].astype(np.float32)

            want_physics = (
                self.include_physics
                and self.physics_mean is not None
                and self.physics_std is not None
            )
            if want_physics:
                physics = _physics_stack(data)
                ds = data["ds"].astype(np.float32)
                dv = data["dv"].astype(np.float32)
                mf_true = data["mf"].astype(np.float32)

            # Augmentacion D4 en espacio CRUDO (antes de normalizar): como el dataset es D4-simetrico
            # con estadisticas por canal identicas para (11)<->(22), permutar componentes es exacto.
            if self.augment_d4:
                op, swap_ab, sign_c = _random_d4()
                z = _sp_apply(z, op)
                fz = _sp_apply(fz, op)
                if want_physics:
                    physics = _apply_tensor_components(_sp_apply(physics, op), swap_ab, sign_c)
                    ds = _sp_apply(ds, op)
                    dv = _sp_apply(dv, op)
                    mf_true = _sp_apply(mf_true, op)

            z_norm = _normalize_minmax(z, self.z_min, self.z_max)
            fz_norm = _normalize_minmax(fz, self.fz_min, self.fz_max)

            item: dict[str, torch.Tensor | str] = {
                "name": path.name,
                "z": torch.from_numpy(np.ascontiguousarray(z_norm)),
                "fz_norm": torch.from_numpy(np.ascontiguousarray(fz_norm)),
                "fz_real": torch.from_numpy(np.ascontiguousarray(fz)),
            }

            if want_physics:
                physics_norm = ((physics - self.physics_mean) / self.physics_std).astype(np.float32)
                item.update(
                    {
                        "physics": torch.from_numpy(np.ascontiguousarray(physics_norm)),
                        "ds": torch.from_numpy(np.ascontiguousarray(ds)),
                        "dv": torch.from_numpy(np.ascontiguousarray(dv)),
                        "mf_true": torch.from_numpy(np.ascontiguousarray(mf_true)),
                    }
                )
            return item
