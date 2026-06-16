"""Lanza un pod de RunPod que entrena TODO el pipeline del TFM y persiste resultados.

Aditivo: no interfiere con el flujo Colab/Drive. Se ejecuta EN LOCAL (Windows) y usa la API
de RunPod para crear el pod, que al arrancar clona el repo y ejecuta runpod/bootstrap.sh.

Requisitos: una RUNPOD_API_KEY en el fichero .env de la raiz del repo.

Ejecucion (sin instalar nada permanente; uv trae 'runpod' de forma efimera):
  uv run --with runpod python runpod/launch.py --list-gpus      # ver IDs de GPU disponibles
  uv run --with runpod python runpod/launch.py                  # crear pod y entrenar TODO
  uv run --with runpod python runpod/launch.py --smoke          # prueba rapida (2 epocas/modelo)
  uv run --with runpod python runpod/launch.py --list           # listar pods
  uv run --with runpod python runpod/launch.py --terminate <id> # terminar pod (parar facturacion)

NOTA: la firma de runpod.create_pod puede variar entre versiones del SDK. Si falla, revisa
`uv run --with runpod python -c "import runpod, inspect; print(inspect.signature(runpod.create_pod))"`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def load_env(env_path: Path) -> None:
    """Parser minimo de .env (KEY=VALUE) -> os.environ. Sin dependencias."""
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    load_env(repo_root / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=os.environ.get("RUNPOD_GPU", "NVIDIA GeForce RTX 4090"),
                        help="gpu_type_id (ver --list-gpus)")
    parser.add_argument("--image", default=os.environ.get(
        "RUNPOD_IMAGE", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"))
    parser.add_argument("--name", default=os.environ.get("RUNPOD_POD_NAME", "tfm-hybrid-pi-ddpm"))
    parser.add_argument("--volume-gb", type=int, default=int(os.environ.get("RUNPOD_VOLUME_GB", "60")))
    parser.add_argument("--disk-gb", type=int, default=int(os.environ.get("RUNPOD_DISK_GB", "30")))
    parser.add_argument("--smoke", action="store_true", help="entrenar solo 2 epocas/modelo")
    parser.add_argument("--no-auto", action="store_true",
                        help="crear el pod pero NO lanzar el entrenamiento (para correr bootstrap a mano)")
    parser.add_argument("--list", action="store_true", help="listar pods existentes")
    parser.add_argument("--list-gpus", action="store_true", help="listar GPUs disponibles")
    parser.add_argument("--terminate", metavar="POD_ID", help="terminar un pod por id")
    args = parser.parse_args()

    try:
        import runpod
    except ImportError:
        print("Falta el SDK. Ejecuta con:  uv run --with runpod python runpod/launch.py ...", file=sys.stderr)
        return 2

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: define RUNPOD_API_KEY en .env (copia .env.example).", file=sys.stderr)
        return 2
    runpod.api_key = api_key

    if args.list_gpus:
        for g in runpod.get_gpus():
            print(f"  {g.get('id')}")
        return 0

    if args.list:
        for p in runpod.get_pods():
            print(f"  {p.get('id')}  {p.get('name')}  [{p.get('desiredStatus')}]")
        return 0

    if args.terminate:
        runpod.terminate_pod(args.terminate)
        print(f"Pod {args.terminate} terminado.")
        return 0

    # ---- construir el comando de arranque del pod ----
    repo_url = os.environ.get("REPO_URL", "https://github.com/DanielArizaGarcia/Hybrid-PI-DDPM.git")
    repo_branch = os.environ.get("REPO_BRANCH", "main")
    smoke = "1" if args.smoke else os.environ.get("SMOKE", "0")

    pod_env = {
        "REPO_URL": repo_url,
        "REPO_BRANCH": repo_branch,
        "WORKDIR": "/workspace",
        "SMOKE": smoke,
    }
    for opt in ("DRIVE_FILE_ID", "DATASET_URL"):
        if os.environ.get(opt):
            pod_env[opt] = os.environ[opt]

    if args.no_auto:
        start_cmd = "bash -lc 'sleep infinity'"
    else:
        start_cmd = (
            "bash -lc 'export PATH=$HOME/.local/bin:$PATH; cd /workspace && "
            "(git clone -b \"$REPO_BRANCH\" \"$REPO_URL\" Hybrid-PI-DDPM || true) && "
            "cd Hybrid-PI-DDPM && git pull || true; "
            "bash runpod/bootstrap.sh 2>&1 | tee /workspace/train.log; "
            "echo \"==== BOOTSTRAP FINISHED ====\"; sleep infinity'"
        )

    print(f"Creando pod '{args.name}' | GPU={args.gpu} | image={args.image}")
    print(f"  repo={repo_url}@{repo_branch} | smoke={smoke} | auto={'no' if args.no_auto else 'si'}")
    pod = runpod.create_pod(
        name=args.name,
        image_name=args.image,
        gpu_type_id=args.gpu,
        cloud_type="SECURE",
        gpu_count=1,
        volume_in_gb=args.volume_gb,
        container_disk_in_gb=args.disk_gb,
        volume_mount_path="/workspace",
        ports="22/tcp,8888/http",
        env=pod_env,
        docker_args=start_cmd,
    )
    pod_id = pod.get("id")
    print("\n================ POD CREADO ================")
    print(f"  id: {pod_id}")
    print("  Estado y SSH: panel de RunPod -> Pods -> Connect")
    print("  Monitorizar entrenamiento (via SSH):  tail -f /workspace/train.log")
    print("  Resultados al terminar:  /workspace/tfm_results.tar.gz  y  /workspace/Hybrid-PI-DDPM/models")
    print(f"  Terminar (parar coste):  uv run --with runpod python runpod/launch.py --terminate {pod_id}")
    print("============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
