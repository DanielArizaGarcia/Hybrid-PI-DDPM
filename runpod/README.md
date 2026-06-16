# Entrenar el TFM en RunPod

Flujo alternativo a Colab/Drive (que la organización deshabilitó). **Aditivo**: no toca el código
del repo ni el flujo Colab; si Colab/Drive vuelven, todo sigue funcionando.

Entrena **todo desde cero** sobre `subset=solid`:
1. `architect_solid` — prior geométrico incondicional (congelado, para E3)
2. `engineer_e0_pb_unet` — baseline
3. `engineer_e0_equino` — operador espectral
4. `engineer_e0_weakrefine` — operador + incertidumbre

---

## Paso 0 — Requisitos (una vez)

1. **Sube estos cambios a GitHub** (el pod clona el repo desde ahí):
   ```powershell
   git -C C:/Hybrid-PI-DDPM add -A
   git -C C:/Hybrid-PI-DDPM commit -m "add: RunPod training (bootstrap, launch, configs solid + E0)"
   git -C C:/Hybrid-PI-DDPM push origin main
   ```
   Si tu repo o rama son otros, ajusta `REPO_URL`/`REPO_BRANCH` en `.env`.

2. **Crea el `.env`** (copia de `.env.example`) y rellena:
   - `RUNPOD_API_KEY` (RunPod → Settings → API Keys)
   - Dataset: lo más cómodo es `DRIVE_FILE_ID` = el id del `processed_image2.zip` en tu Drive
     compartido como "cualquiera con el enlace". (El id es la parte de la URL
     `drive.google.com/file/d/<ESTE_ID>/view`.) Alternativas: `DATASET_URL`, o subir el zip a mano.

---

## Opción 1 — Automático (API, recomendado)

Desde la raíz del repo en tu PC:

```powershell
# (opcional) ver IDs de GPU disponibles y elegir
uv run --with runpod python runpod/launch.py --list-gpus

# prueba rápida primero (2 épocas/modelo, ~minutos) para validar que todo corre
uv run --with runpod python runpod/launch.py --smoke

# run completo (entrena los 4 modelos)
uv run --with runpod python runpod/launch.py
```

El pod, al arrancar, clona el repo, instala `uv`, descarga el dataset, lo verifica y entrena todo,
guardando el log en `/workspace/train.log`. `launch.py` te imprime el `id` del pod y los comandos
para monitorizar/terminar.

**Monitorizar** (SSH del panel de RunPod → Connect):
```bash
tail -f /workspace/train.log
```

**Al terminar**, descarga resultados y **termina el pod** para no seguir pagando:
```powershell
uv run --with runpod python runpod/launch.py --terminate <POD_ID>
```

---

## Opción 2 — Manual (sin API, desde el panel de RunPod)

1. En runpod.io → **Deploy** un pod: plantilla **PyTorch** (CUDA 12.x), una GPU (RTX 4090/A5000
   sobran), y **Network Volume** montado en `/workspace` (para que persistan los checkpoints).
2. Abre el **Web Terminal** del pod y ejecuta:
   ```bash
   cd /workspace
   git clone -b main https://github.com/DanielArizaGarcia/Hybrid-PI-DDPM.git Hybrid-PI-DDPM
   cd Hybrid-PI-DDPM
   export DRIVE_FILE_ID="<id_del_zip_en_drive>"   # o DATASET_URL=...
   bash runpod/bootstrap.sh 2>&1 | tee /workspace/train.log
   ```
   Prueba rápida: `SMOKE=1 bash runpod/bootstrap.sh`.

---

## Salidas

Dentro del pod (persisten si `/workspace` es un Network Volume):
- `/workspace/Hybrid-PI-DDPM/models/<role>/<timestamp>_<config>/best.pt` — checkpoints
- `/workspace/Hybrid-PI-DDPM/artifacts/...` — curvas y figuras
- `/workspace/Hybrid-PI-DDPM/mlruns/` — tracking MLflow
- `/workspace/tfm_results.tar.gz` — todo empaquetado para descargar

Descarga el `.tar.gz` desde el explorador de archivos del panel de RunPod, o con `runpodctl receive`.

---

## Notas

- **No puedo lanzar RunPod por ti** (necesita tu API key y red): estos scripts están escritos con
  cuidado pero **haz primero `--smoke`** para validar antes del run completo.
- La firma de `runpod.create_pod` cambia entre versiones del SDK. Si `launch.py` falla al crear el
  pod, usa la **Opción 2 (manual)** — `bootstrap.sh` es independiente del SDK.
- El dataset corrupto no aborta el run: `verify_dataset.py` mueve cualquier `.npz` ilegible a
  `processed_image2/_corrupt/` antes de entrenar.
- E3 (muestreo guiado) se lanza después, eligiendo el checkpoint de engineer, con
  `sample_guided.py` (ver `docs/PLAN_TFM_DANI.md`).
