"""R3 (Colab) - Corrida FINAL del guiado adaptativo: generacion + calidad/diversidad + bootstrap.

Autocontenido y Colab-ready: detecta Colab, lee modelos del Drive, ENTRENA la VAE inline (si no esta),
genera las 4 arms (no_guide / pb_bell / a3_bell / a3_gate) a STEPS=250 N=256, juzga la calidad con a3,
mide diversidad (hull/Vendi latente + pixel), corre bootstrap (IC95%), y GUARDA TODO en el Drive.

En Colab (tras montar Drive y descomprimir el dataset en /content/Datos/processed_image2):
  !cd <repo> && git pull
  !uv run --with scipy python scripts/r3_colab.py
Local:
  uv run --with scipy python scripts/r3_colab.py
"""
from __future__ import annotations
import sys, os, math, time, glob
os.environ["MPLBACKEND"] = "Agg"   # Colab fija el backend inline; forzar Agg ANTES de importar matplotlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tfm_shells.models.factory import build_scheduler, build_unet
from tfm_shells.utils.physics import compute_membrane_factor_from_prediction

# ---------------- paths: Colab (Drive) o local ----------------
if Path("/content/drive/MyDrive").exists():
    ROOT = Path("/content/drive/MyDrive/Hybrid-PI-DDPM-runtime")
    DATA = Path("/content/Datos/processed_image2")
    OUT = ROOT / "artifacts" / "r3_adaptive_final"
else:
    ROOT = Path(r"G:\Mi unidad\Hybrid-PI-DDPM-runtime")
    DATA = Path(r"C:\Datos\processed_image2")
    OUT = Path(r"C:\Hybrid-PI-DDPM\artifacts\r3_adaptive_final")
MENG = ROOT / "models" / "engineer"
MARCH = ROOT / "models" / "architect"
ARCH = MARCH / "20260620_105510_colab_e5_architect_solid" / "best.pt"
PBUNET = MENG / "20260616_165442_engineer_e0_pb_unet" / "best.pt"
A3 = MENG / "20260620_113638_colab_e5_uq_a3_weakrefine_full" / "best.pt"
VAE_CK = OUT / "vae_solid.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = int(os.environ.get("R3_N", 256)); BATCH = 8
# STEPS = pasos de muestreo. Fiel al entrenamiento = T=1000 (= sample_guided.yaml). 100/250 son
# aproximaciones aceleradas (subconjunto de los 1000) y dan resultados distintos -> usar 1000.
STEPS = int(os.environ.get("R3_STEPS", 1000)); SCALE = 600.0
W_MAX, BELL_PEAK, BELL_WIDTH, GRAD_CLIP, SEED = 1.0, 0.5, 0.22, 20.0, 7
TAU_A3 = 1.63
B_BOOT = 3000


def log(*a): print(*a, flush=True)
def bell_env(step, total):
    r = step / (total - 1) if total > 1 else 0.0
    return W_MAX * math.exp(-((r - BELL_PEAK) ** 2) / (2 * BELL_WIDTH ** 2))
def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_unet(ck["model_config"]); m.load_state_dict(ck["model_state_dict"]); m.to(DEVICE).eval()
    return m, ck
def renorm(x, a_min, a_max, e_min, e_max):
    xr = ((x + 1) / 2) * (a_max - a_min) + a_min
    return 2 * ((xr - e_min) / (e_max - e_min + 1e-8)) - 1


# ---------------- VAE (entrena inline si no esta) ----------------
class VAE(nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.Conv2d(128, 128, 4, 2, 1), nn.ReLU())
        self.fc_mu = nn.Linear(128 * 4 * 4, zdim); self.fc_lv = nn.Linear(128 * 4 * 4, zdim)
        self.fc_d = nn.Linear(zdim, 128 * 4 * 4)
        self.dec = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                                 nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh())
    def encode(self, x):
        h = self.enc(x).flatten(1); return self.fc_mu(h), self.fc_lv(h)
    def forward(self, x):
        mu, lv = self.encode(x); z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.dec(self.fc_d(z).view(-1, 128, 4, 4)), mu, lv


def get_vae(zmin, zmax):
    vae = VAE().to(DEVICE)
    if VAE_CK.exists():
        vae.load_state_dict(torch.load(VAE_CK, map_location=DEVICE)); log("VAE cargada de", VAE_CK); return vae.eval()
    log("Entrenando VAE inline sobre solid...")
    files = sorted(f for f in glob.glob(str(DATA / "*.npz")) if "hole" not in os.path.basename(f))
    from sklearn.model_selection import train_test_split
    tr, _ = train_test_split(files, test_size=0.20, random_state=42, shuffle=True)
    Z = np.stack([np.load(f)["z"][0].astype(np.float32) for f in tr])
    X = torch.from_numpy(2 * (Z - zmin) / (zmax - zmin) - 1).unsqueeze(1)
    torch.manual_seed(SEED); opt = torch.optim.Adam(vae.parameters(), 1e-3); n = X.shape[0]
    for ep in range(40):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            xb = X[perm[i:i + 64]].to(DEVICE)
            rec, mu, lv = vae(xb)
            loss = ((rec - xb) ** 2).mean() + 1e-3 * (-0.5 * (1 + lv - mu ** 2 - lv.exp()).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    OUT.mkdir(parents=True, exist_ok=True); torch.save(vae.state_dict(), VAE_CK)
    log("VAE entrenada y guardada en", VAE_CK); return vae.eval()


# ---------------- generacion ----------------
def gen(arm, architect, a_stats, engineer, e_stats, scheduler, n, seed):
    a_min, a_max = float(a_stats["z_min"]), float(a_stats["z_max"])
    pm = ps = None; e_min = e_max = 0.0
    if e_stats is not None:
        pm = torch.tensor(e_stats["physics_mean"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        ps = torch.tensor(e_stats["physics_std"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        e_min, e_max = float(e_stats["z_min"]), float(e_stats["z_max"])
    ts = scheduler.timesteps; total = len(ts); zs = []
    for start in range(0, n, BATCH):
        bs = min(BATCH, n - start)
        g0 = torch.Generator(device=DEVICE).manual_seed(seed + start)
        x = torch.randn((bs, 1, 64, 64), generator=g0, device=DEVICE)
        fzc = torch.zeros((bs, 1, 64, 64), device=DEVICE)
        pmb = pm.expand(bs, -1, -1, -1) if pm is not None else None
        psb = ps.expand(bs, -1, -1, -1) if ps is not None else None
        for i, t in enumerate(ts):
            tb = torch.full((bs,), int(t.item()), device=DEVICE, dtype=torch.long)
            with torch.no_grad():
                v = architect(x, tb).sample
            if arm != "no_guide":
                xr = x.detach().clone().requires_grad_(True)
                xr_e = renorm(xr, a_min, a_max, e_min, e_max)
                out = engineer(torch.cat([xr_e, fzc], dim=1), tb)
                mf = compute_membrane_factor_from_prediction(out.sample, pmb, psb).mean(dim=(1, 2, 3))
                g = torch.autograd.grad(((1.0 - mf) ** 2).sum(), xr)[0]
                abar = scheduler.alphas_cumprod[t].to(DEVICE)
                env = SCALE * W_MAX * bell_env(i, total)
                if arm == "a3_gate":
                    with torch.no_grad():
                        sig2 = TAU_A3 * torch.exp(out.log_variance[:, :1])
                        inv = 1.0 / (sig2 + 1e-6)
                        gate = torch.clamp(inv / inv.mean(dim=(2, 3), keepdim=True), 0.2, 5.0)
                    grad = torch.clamp(env * gate * g, -GRAD_CLIP, GRAD_CLIP)
                else:
                    grad = torch.clamp(env * g, -GRAD_CLIP, GRAD_CLIP)
                v = v + torch.sqrt(1.0 - abar) * grad
                del out, xr, mf, g
            x = scheduler.step(v, t, x).prev_sample
        zs.append(x.detach())
    z = torch.cat(zs, 0)
    a_min, a_max = float(a_stats["z_min"]), float(a_stats["z_max"])
    zr = ((z.cpu().numpy() + 1) / 2) * (a_max - a_min) + a_min
    return zr[:, 0], z   # (n,64,64) real (sin canal), y z normalizado (n,1,64,64)


def judge_mf(z_norm_arch, a_stats, judge, j_stats):
    a_min, a_max = float(a_stats["z_min"]), float(a_stats["z_max"])
    j_min, j_max = float(j_stats["z_min"]), float(j_stats["z_max"])
    pm = torch.tensor(j_stats["physics_mean"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    ps = torch.tensor(j_stats["physics_std"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mfs = []
    with torch.no_grad():
        for s in range(0, z_norm_arch.shape[0], BATCH):
            zb = z_norm_arch[s:s + BATCH].to(DEVICE)
            ze = renorm(zb, a_min, a_max, j_min, j_max)
            fz0 = torch.zeros((zb.shape[0], 1, 64, 64), device=DEVICE)
            tb0 = torch.zeros(zb.shape[0], device=DEVICE, dtype=torch.long)
            pred = judge(torch.cat([ze, fz0], dim=1), tb0).sample
            mf = compute_membrane_factor_from_prediction(pred, pm.expand(zb.shape[0], -1, -1, -1),
                                                         ps.expand(zb.shape[0], -1, -1, -1)).mean(dim=(1, 2, 3))
            mfs.append(mf.cpu().numpy())
    return np.concatenate(mfs)


# ---------------- metricas + bootstrap ----------------
def _pd2(X): sq = (X * X).sum(1); return np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
def vendi(X):
    d2 = _pd2(X.astype(np.float64)); iu = np.triu_indices(len(X), 1)
    s2 = np.median(d2[iu]) + 1e-12; K = np.exp(-d2 / (2 * s2)) / len(X)
    ev = np.linalg.eigvalsh(K); ev = ev[ev > 1e-12]; ev /= ev.sum()
    return float(np.exp(-(ev * np.log(ev)).sum()))
def hull_area(L):
    from scipy.spatial import ConvexHull
    from sklearn.decomposition import PCA
    P = PCA(2).fit_transform(L)
    try: return float(ConvexHull(P).volume)
    except Exception: return float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"device={DEVICE} N={N} STEPS={STEPS} | OUT={OUT}")
    for p in (ARCH, PBUNET, A3):
        if not p.exists(): log("FALTA:", p); return
    architect, ack = load(ARCH); a_stats = ack["normalization_stats"]
    pb, pbk = load(PBUNET); a3, a3k = load(A3)
    scheduler = build_scheduler(ack["model_config"]); scheduler.set_timesteps(STEPS, device=DEVICE)
    zmin, zmax = float(a_stats["z_min"]), float(a_stats["z_max"])
    vae = get_vae(zmin, zmax)

    ARMS = [("no_guide", None, None), ("pb_bell", pb, pbk["normalization_stats"]),
            ("a3_bell", a3, a3k["normalization_stats"]), ("a3_gate", a3, a3k["normalization_stats"])]
    geoms, znorm = {}, {}
    for arm, eng, est in ARMS:
        t0 = time.time(); log(f"  generando {arm}...")
        zr, zn = gen(arm, architect, a_stats, eng, est, scheduler, N, SEED)
        geoms[arm] = zr; znorm[arm] = zn; log(f"  generado {arm} ({time.time()-t0:.0f}s)")

    # guardar las geometrias YA (lo caro), antes del post-proceso, por si algo falla luego
    np.savez(OUT / "r3_geoms.npz", **{f"Z_{a}": geoms[a] for a in geoms})
    log("geometrias guardadas:", OUT / "r3_geoms.npz")

    mfj = {a: judge_mf(znorm[a], a_stats, a3, a3k["normalization_stats"]) for a in geoms}
    lat = {}
    with torch.no_grad():
        for a in geoms:
            zn = 2 * (geoms[a] - zmin) / (zmax - zmin) - 1
            mu, _ = vae.encode(torch.from_numpy(zn).unsqueeze(1).float().to(DEVICE)); lat[a] = mu.cpu().numpy()

    np.savez(OUT / "r3_geoms.npz", **{f"Z_{a}": geoms[a] for a in geoms}, **{f"mfj_{a}": mfj[a] for a in geoms})

    # tabla
    rows = {}
    log(f"\n{'arm':9s} | mf P(>.9) | hull medlat Vlat Vpx")
    for a in geoms:
        Z, L, mf = geoms[a], lat[a], mfj[a]
        rows[a] = dict(mf=float(mf.mean()), p90=float((mf > 0.9).mean()), hull=hull_area(L),
                       vlat=vendi(L), vpx=vendi(Z.reshape(len(Z), -1)))
        log(f"{a:9s} | {rows[a]['mf']:.3f} {rows[a]['p90']:.2f} | {rows[a]['hull']:6.2f} {rows[a]['vlat']:5.2f} {rows[a]['vpx']:5.2f}")

    # bootstrap
    n = N; rng = np.random.default_rng(0); METR = ["mf", "p90", "hull", "vlat", "vpx"]
    def mval(a, idx, m):
        if m == "mf": return float(mfj[a][idx].mean())
        if m == "p90": return float((mfj[a][idx] > 0.9).mean())
        if m == "hull": return hull_area(lat[a][idx])
        if m == "vlat": return vendi(lat[a][idx])
        if m == "vpx": return vendi(geoms[a].reshape(n, -1)[idx])
    boot = {a: {m: np.empty(B_BOOT) for m in METR} for a in geoms}
    log(f"\nbootstrap B={B_BOOT}...")
    for a in geoms:
        for b in range(B_BOOT):
            idx = rng.integers(0, n, n)
            for m in METR: boot[a][m][b] = mval(a, idx, m)

    with (OUT / "r3_final_summary.md").open("w", encoding="utf-8") as fh:
        fh.write(f"# R3 FINAL (N={N}, STEPS={STEPS}) - guiado adaptativo\n\n")
        fh.write("| arm | mf(a3) | P(mf>0.9) | hull | Vendi_lat | Vendi_px |\n|---|---|---|---|---|---|\n")
        for a in geoms:
            r = rows[a]; fh.write(f"| {a} | {r['mf']:.3f} | {r['p90']:.2f} | {r['hull']:.2f} | {r['vlat']:.2f} | {r['vpx']:.2f} |\n")
        fh.write(f"\n## Bootstrap (B={B_BOOT}, IC95%, * = excluye 0)\n\n")
        fh.write("| comparacion | metrica | dif | IC95% | P(A>B) |\n|---|---|---|---|---|\n")
        names = {"mf": "mf", "p90": "P(mf>0.9)", "hull": "hull", "vlat": "Vendi_lat", "vpx": "Vendi_px"}
        for A, Bb in [("a3_gate", "a3_bell"), ("a3_gate", "pb_bell"), ("a3_gate", "no_guide")]:
            for m in METR:
                d = boot[A][m] - boot[Bb][m]; lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
                sig = "*" if (lo > 0 or hi < 0) else ""
                fh.write(f"| {A} vs {Bb} | {names[m]} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}]{sig} | {(d>0).mean():.3f} |\n")
                log(f"  {A} vs {Bb} {names[m]:10s} dif={d.mean():+.3f} IC[{lo:+.3f},{hi:+.3f}]{sig} P={float((d>0).mean()):.3f}")

    # frontera
    plt.figure(figsize=(7.5, 6)); col = {"no_guide": "#7f7f7f", "pb_bell": "#000", "a3_bell": "#1f77b4", "a3_gate": "#d62728"}
    for a in geoms:
        plt.scatter(rows[a]["hull"], rows[a]["mf"], s=120, color=col[a], zorder=3)
        plt.annotate(a, (rows[a]["hull"], rows[a]["mf"]), textcoords="offset points", xytext=(8, 4))
    plt.xlabel("diversidad: hull area (VAE)"); plt.ylabel("calidad: mf (juez a3)")
    plt.title(f"R3 FINAL (N={N}, STEPS={STEPS}) - frontera calidad/diversidad"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_r3_final_pareto.png", dpi=160); plt.close()
    log(f"\nTODO guardado en {OUT}\nR3 FINAL COMPLETO.")


if __name__ == "__main__":
    main()
