"""Verifica la integridad del dataset de .npz y mueve los corruptos a cuarentena.

Aditivo: NO toca el codigo del repo. Se ejecuta antes de entrenar en RunPod para que un
unico .npz corrupto (p. ej. una descarga truncada) no aborte un run de varias horas, ya que
`build_dataset_index` hace `np.load` de cada fichero y lanzaria EOFError.

Uso:
  uv run python runpod/verify_dataset.py --dir ../Datos/processed_image2
  uv run python runpod/verify_dataset.py --dir /workspace/Datos/processed_image2 --quarantine

Salida: cuenta de validos / corruptos. Con --quarantine, mueve los corruptos a
<dir>/_corrupt/ para que el indexador no los vea. Devuelve codigo !=0 si quedan corruptos
en el directorio (sin --quarantine) para poder cortar el pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Carpeta con los .npz")
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="Mover los .npz corruptos a <dir>/_corrupt/ en vez de solo reportarlos",
    )
    parser.add_argument(
        "--keys",
        default="z,fz,mf",
        help="Claves minimas que deben poder leerse (coma-separadas)",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"ERROR: no existe la carpeta {root}", file=sys.stderr)
        return 2

    required_keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    files = sorted(p for p in root.glob("*.npz"))
    print(f"Verificando {len(files)} ficheros .npz en {root} ...")

    corrupt: list[tuple[Path, str]] = []
    for path in files:
        try:
            with np.load(path) as data:
                for key in required_keys:
                    _ = data[key].shape
        except Exception as error:  # noqa: BLE001
            corrupt.append((path, f"{type(error).__name__}: {error}"))

    valid = len(files) - len(corrupt)
    print(f"Validos: {valid} | Corruptos: {len(corrupt)}")

    if corrupt:
        for path, err in corrupt:
            print(f"  CORRUPTO {path.name}: {err}")
        if args.quarantine:
            qdir = root / "_corrupt"
            qdir.mkdir(exist_ok=True)
            for path, _ in corrupt:
                path.rename(qdir / path.name)
            print(f"Movidos {len(corrupt)} ficheros a {qdir}")
            return 0
        return 1

    print("Dataset integro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
