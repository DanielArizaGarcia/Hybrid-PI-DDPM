#!/usr/bin/env bash
# ==============================================================================
# bootstrap.sh - Entrena TODO el pipeline del TFM dentro de un pod de RunPod.
#
# Aditivo: NO interfiere con el flujo Colab/Drive. Pensado para Linux + GPU.
#
# Entrena (subset=solid): Architect incondicional + 3 engineers (E0).
#   1. architect_solid          (prior geometrico congelado para E3)
#   2. engineer_e0_pb_unet       (baseline)
#   3. engineer_e0_equino        (operador espectral)
#   4. engineer_e0_weakrefine    (operador + incertidumbre)
#
# Variables de entorno (todas opcionales salvo donde se indica):
#   REPO_URL       URL del repo git (def: https://github.com/DanielArizaGarcia/Hybrid-PI-DDPM.git)
#   REPO_BRANCH    rama a clonar (def: main)
#   WORKDIR        raiz de trabajo persistente (def: /workspace)
#   DRIVE_FILE_ID  id de Google Drive del processed_image2.zip (para gdown)  [opcion A]
#   DATASET_URL    URL directa de descarga del zip (wget/curl)               [opcion B]
#                  (si no se da ninguna, se espera $WORKDIR/Datos/processed_image2.zip ya subido) [opcion C]
#   SMOKE          si =1, entrena solo 2 epocas por modelo (prueba rapida)
#
# Salidas (persisten si WORKDIR esta en un Network Volume):
#   $WORKDIR/Hybrid-PI-DDPM/models, artifacts, mlruns
#   $WORKDIR/tfm_results.tar.gz   (empaquetado final para descargar)
# ==============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/DanielArizaGarcia/Hybrid-PI-DDPM.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORKDIR="${WORKDIR:-/workspace}"
SMOKE="${SMOKE:-0}"

log() { echo -e "\n\033[1;36m[bootstrap] $*\033[0m"; }

# ------------------------------------------------------------------ 1. uv
if ! command -v uv >/dev/null 2>&1; then
  log "Instalando uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# ------------------------------------------------------------------ 2. repo
mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ -d "Hybrid-PI-DDPM/.git" ]; then
  log "Repo existente -> git pull ($REPO_BRANCH)"
  cd Hybrid-PI-DDPM && git fetch origin && git checkout "$REPO_BRANCH" && git pull origin "$REPO_BRANCH"
else
  log "Clonando $REPO_URL ($REPO_BRANCH)"
  git clone -b "$REPO_BRANCH" "$REPO_URL" Hybrid-PI-DDPM
  cd Hybrid-PI-DDPM
fi
REPO_DIR="$WORKDIR/Hybrid-PI-DDPM"

# ------------------------------------------------------------------ 3. entorno
log "uv sync ..."
uv sync
uv run python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"

# ------------------------------------------------------------------ 4. dataset
DATA_DIR="$WORKDIR/Datos/processed_image2"
mkdir -p "$WORKDIR/Datos"
if [ -d "$DATA_DIR" ] && [ "$(ls -1 "$DATA_DIR"/*.npz 2>/dev/null | wc -l)" -gt 0 ]; then
  log "Dataset ya presente en $DATA_DIR ($(ls -1 "$DATA_DIR"/*.npz | wc -l) .npz)"
else
  ZIP="$WORKDIR/Datos/processed_image2.zip"
  if [ -n "${DRIVE_FILE_ID:-}" ]; then
    log "Descargando dataset de Google Drive (id=$DRIVE_FILE_ID) con gdown ..."
    uv run --with gdown python -m gdown "$DRIVE_FILE_ID" -O "$ZIP"
  elif [ -n "${DATASET_URL:-}" ]; then
    log "Descargando dataset de $DATASET_URL ..."
    curl -L "$DATASET_URL" -o "$ZIP"
  elif [ -f "$ZIP" ]; then
    log "Usando zip ya subido en $ZIP"
  else
    echo "ERROR: no hay dataset. Define DRIVE_FILE_ID o DATASET_URL, o sube el zip a $ZIP" >&2
    exit 1
  fi
  log "Descomprimiendo $ZIP -> $WORKDIR/Datos ..."
  uv run python - "$ZIP" "$WORKDIR/Datos" <<'PY'
import sys, zipfile, os
zip_path, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as z:
    for m in z.namelist():
        if m.startswith("__MACOSX/") or m.endswith("/"):
            continue
        z.extract(m, dest)
print("extraido")
PY
fi

# ------------------------------------------------------------------ 5. verificar dataset
log "Verificando integridad del dataset (mueve corruptos a cuarentena) ..."
uv run python runpod/verify_dataset.py --dir "$DATA_DIR" --quarantine

# ------------------------------------------------------------------ 6. (opcional) smoke
CFG_ARCH="configs/architect_solid.yaml"
CFG_PB="configs/engineer_e0_pb_unet.yaml"
CFG_EQ="configs/engineer_e0_equino.yaml"
CFG_WR="configs/engineer_e0_weakrefine.yaml"

if [ "$SMOKE" = "1" ]; then
  log "MODO SMOKE: copiando configs con epochs=2 ..."
  mkdir -p configs/_smoke
  for src in "$CFG_ARCH" "$CFG_PB" "$CFG_EQ" "$CFG_WR"; do
    dst="configs/_smoke/$(basename "$src")"
    sed -E 's/^( *epochs: ).*/\12/; s/^( *warmup_epochs: ).*/\11/; s/^( *sample_every_n_epochs: ).*/\11/' "$src" > "$dst"
  done
  CFG_ARCH="configs/_smoke/$(basename "$CFG_ARCH")"
  CFG_PB="configs/_smoke/$(basename "$CFG_PB")"
  CFG_EQ="configs/_smoke/$(basename "$CFG_EQ")"
  CFG_WR="configs/_smoke/$(basename "$CFG_WR")"
fi

# ------------------------------------------------------------------ 7. ENTRENAR TODO
log "=== 1/4 Architect (solid) ==="
uv run python train_architect.py --config "$CFG_ARCH"
log "=== 2/4 Engineer parallel_pb_unet ==="
uv run python train_engineer.py --config "$CFG_PB"
log "=== 3/4 Engineer equino ==="
uv run python train_engineer.py --config "$CFG_EQ"
log "=== 4/4 Engineer shell_weakrefine_operator ==="
uv run python train_engineer.py --config "$CFG_WR"

# ------------------------------------------------------------------ 8. empaquetar resultados
log "Empaquetando resultados en $WORKDIR/tfm_results.tar.gz ..."
tar -czf "$WORKDIR/tfm_results.tar.gz" -C "$REPO_DIR" models artifacts mlruns 2>/dev/null || true

log "ENTRENAMIENTO COMPLETO. Checkpoints en $REPO_DIR/models, figuras en $REPO_DIR/artifacts."
log "Descarga $WORKDIR/tfm_results.tar.gz antes de TERMINAR el pod."
