"""R4b - Barrido de FUERZA de guiado (gamma) con equino+sigma2 en SOLID, PROTOCOLO COMPLETO:
N=256, 1000 pasos (el regimen de entrenamiento), bell fija, seed 7 pareada con R3b.

Reusa de R3b los brazos gamma=0 (no_guide) y gamma=600 (eq_bell), generados con el MISMO
protocolo, de modo que las cifras del barrido cuadran exactamente con la tabla principal.
Genera gamma in {100, 250, 500, 1000, 2000} y anade IC95% bootstrap para mf y hull.

Uso:  .venv\\Scripts\\python.exe scripts/r4b_guidance_strength_1000.py
"""
from __future__ import annotations
import sys, os, math, time
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from tfm_shells.models.factory import build_scheduler, build_unet
from tfm_shells.utils.physics import compute_membrane_factor_from_prediction

if Path("/content/drive/MyDrive").exists():           # Colab
    R = Path("/content/drive/MyDrive/Hybrid-PI-DDPM-runtime")
    IMG = R / "artifacts" / "r4b_guidance_strength_1000"   # la figura tambien a Drive
else:                                                   # local
    R = Path(r"G:\Mi unidad\Hybrid-PI-DDPM-runtime")
    IMG = Path(r"C:\Hybrid-PI-DDPM\memoria_tfm\images")
ARCH = R / "models" / "architect" / "20260620_105510_colab_e5_architect_solid" / "best.pt"
ENG = R / "models" / "engineer" / "20260625_180430_colab_e6_equino_unc_solid" / "best.pt"
VAE_CK = R / "artifacts" / "r4b_deps" / "vae_solid.pt"
R3B = R / "artifacts" / "r4b_deps" / "r3b_geoms.npz"
OUTDIR = R / "artifacts" / "r4b_guidance_strength_1000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N, STEPS, BATCH = 256, 1000, 8
GAMMAS_NEW = [100, 250, 500, 1000, 2000]
W_MAX, BELL_PEAK, BELL_WIDTH, GRAD_CLIP, SEED = 1.0, 0.5, 0.22, 20.0, 7
ZMIN, ZMAX = 0.0, 7.850908857450063
B_BOOT = 2000


def log(*a): print(*a, flush=True)
def bell_env(step, total):
    r = step / (total - 1) if total > 1 else 0.0
    return W_MAX * math.exp(-((r - BELL_PEAK) ** 2) / (2 * BELL_WIDTH ** 2))
def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = build_unet(ck["model_config"]); m.load_state_dict(ck["model_state_dict"]); m.to(DEVICE).eval()
    return m, ck
def renorm(x, a_min, a_max, e_min, e_max):
    xr = ((x + 1) / 2) * (a_max - a_min) + a_min
    return 2 * ((xr - e_min) / (e_max - e_min + 1e-8)) - 1


class VAE(nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.Conv2d(128, 128, 4, 2, 1), nn.ReLU())
        self.fc_mu = nn.Linear(128 * 4 * 4, zdim); self.fc_lv = nn.Linear(128 * 4 * 4, zdim)
    def encode(self, x):
        h = self.enc(x).flatten(1); return self.fc_mu(h), self.fc_lv(h)


def gen(architect, a_stats, engineer, e_stats, scheduler, gamma, n, seed):
    a_min, a_max = float(a_stats["z_min"]), float(a_stats["z_max"])
    pm = torch.tensor(e_stats["physics_mean"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    ps = torch.tensor(e_stats["physics_std"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    e_min, e_max = float(e_stats["z_min"]), float(e_stats["z_max"])
    ts = scheduler.timesteps; total = len(ts); zs = []
    for start in range(0, n, BATCH):
        bs = min(BATCH, n - start)
        g0 = torch.Generator(device=DEVICE).manual_seed(seed + start)
        x = torch.randn((bs, 1, 64, 64), generator=g0, device=DEVICE)
        fzc = torch.zeros((bs, 1, 64, 64), device=DEVICE)
        pmb = pm.expand(bs, -1, -1, -1); psb = ps.expand(bs, -1, -1, -1)
        for i, t in enumerate(ts):
            tb = torch.full((bs,), int(t.item()), device=DEVICE, dtype=torch.long)
            with torch.no_grad():
                v = architect(x, tb).sample
                abar = scheduler.alphas_cumprod[t].to(DEVICE)
            xr = x.detach().clone().requires_grad_(True)
            xr_e = renorm(xr, a_min, a_max, e_min, e_max)
            out = engineer(torch.cat([xr_e, fzc], dim=1), tb)
            mf = compute_membrane_factor_from_prediction(out.sample, pmb, psb).mean(dim=(1, 2, 3))
            gr = torch.autograd.grad(((1.0 - mf) ** 2).sum(), xr)[0]
            env = gamma * W_MAX * bell_env(i, total)
            grad = torch.clamp(env * gr, -GRAD_CLIP, GRAD_CLIP)
            v = v + torch.sqrt(1.0 - abar) * grad
            del out, xr, mf, gr
            x = scheduler.step(v, t, x).prev_sample
        zs.append(x.detach())
    z = torch.cat(zs, 0)
    return (((z.cpu().numpy() + 1) / 2) * (a_max - a_min) + a_min)[:, 0]


def judge_mf(zr, a_stats, judge, j_stats):
    a_min, a_max = float(a_stats["z_min"]), float(a_stats["z_max"])
    j_min, j_max = float(j_stats["z_min"]), float(j_stats["z_max"])
    pm = torch.tensor(j_stats["physics_mean"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    ps = torch.tensor(j_stats["physics_std"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    zna = torch.from_numpy(2 * (zr - a_min) / (a_max - a_min) - 1).unsqueeze(1).float()
    mfs = []
    with torch.no_grad():
        for s in range(0, zna.shape[0], BATCH):
            zb = zna[s:s + BATCH].to(DEVICE)
            ze = renorm(zb, a_min, a_max, j_min, j_max)
            fz0 = torch.zeros((zb.shape[0], 1, 64, 64), device=DEVICE)
            tb0 = torch.zeros(zb.shape[0], device=DEVICE, dtype=torch.long)
            pred = judge(torch.cat([ze, fz0], dim=1), tb0).sample
            mf = compute_membrane_factor_from_prediction(pred, pm.expand(zb.shape[0], -1, -1, -1),
                                                         ps.expand(zb.shape[0], -1, -1, -1)).mean(dim=(1, 2, 3))
            mfs.append(mf.cpu().numpy())
    return np.concatenate(mfs)


def latents(vae, geoms):
    zna = torch.from_numpy(2 * (geoms - ZMIN) / (ZMAX - ZMIN) - 1).unsqueeze(1).float().to(DEVICE)
    with torch.no_grad():
        mu, _ = vae.encode(zna)
    return mu.cpu().numpy()


def hull_area(L):
    P = PCA(2).fit_transform(L)
    try: return float(ConvexHull(P).volume)
    except Exception: return float("nan")


def boot_ci(vals_fn, n, B=B_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    est = np.empty(B)
    for b in range(B):
        est[b] = vals_fn(rng.integers(0, n, n))
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True); IMG.mkdir(parents=True, exist_ok=True)
    log(f"device={DEVICE} N={N} STEPS={STEPS} gammas_nuevos={GAMMAS_NEW} (+0 y 600 reusados de R3b)")
    architect, ack = load(ARCH); a_stats = ack["normalization_stats"]
    engineer, eck = load(ENG); e_stats = eck["normalization_stats"]
    scheduler = build_scheduler(ack["model_config"]); scheduler.set_timesteps(STEPS, device=DEVICE)
    vae = VAE().to(DEVICE).eval(); vae.load_state_dict(torch.load(VAE_CK, map_location=DEVICE), strict=False)

    geoms, mfs = {}, {}
    r3b = np.load(R3B)
    geoms[0] = r3b["Z_no_guide"]; mfs[0] = r3b["mfeq_no_guide"]
    geoms[600] = r3b["Z_eq_bell"]; mfs[600] = r3b["mfeq_eq_bell"]
    log("  reusados gamma=0 y gamma=600 de R3b (mismo protocolo, mismas semillas)")

    ckpt = OUTDIR / "r4b_geoms.npz"
    if ckpt.exists():   # reanudar: cargar gammas ya generados
        prev = np.load(ckpt)
        for g in GAMMAS_NEW:
            if f"Z_g{g}" in prev.files:
                geoms[g] = prev[f"Z_g{g}"]; mfs[g] = prev[f"mf_g{g}"]
                log(f"  reanudado gamma={g} desde checkpoint")

    for gamma in GAMMAS_NEW:
        if gamma in geoms:
            continue
        t0 = time.time()
        zr = gen(architect, a_stats, engineer, e_stats, scheduler, gamma, N, SEED)
        geoms[gamma] = zr
        mfs[gamma] = judge_mf(zr, a_stats, engineer, e_stats)
        np.savez(ckpt, **{f"Z_g{g}": geoms[g] for g in geoms},
                 **{f"mf_g{g}": mfs[g] for g in mfs})
        log(f"  gamma={gamma:5d} generado y juzgado ({time.time()-t0:.0f}s)  [guardado]")

    lat = {g: latents(vae, geoms[g]) for g in geoms}
    rows = []
    log(f"\n{'gamma':>6s} | mf [IC95]           | P>.8  P>.9 | hull [IC95]")
    for g in sorted(geoms):
        mf = mfs[g]; L = lat[g]
        mf_lo, mf_hi = boot_ci(lambda idx, m=mf: float(m[idx].mean()), N)
        h = hull_area(L)
        h_lo, h_hi = boot_ci(lambda idx, LL=L: hull_area(LL[idx]), N, B=1000)
        rows.append((g, float(mf.mean()), mf_lo, mf_hi, float((mf > 0.8).mean()),
                     float((mf > 0.9).mean()), h, h_lo, h_hi))
        log(f"{g:6d} | {mf.mean():.3f} [{mf_lo:.3f},{mf_hi:.3f}] | {(mf>0.8).mean():.2f}  {(mf>0.9).mean():.2f} | {h:5.1f} [{h_lo:.1f},{h_hi:.1f}]")

    gam = [r[0] for r in rows]; mfm = [r[1] for r in rows]
    mf_err = [[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]]
    hl = [r[6] for r in rows]
    h_err = [[r[6] - r[7] for r in rows], [r[8] - r[6] for r in rows]]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax2 = ax[0].twinx()
    l1 = ax[0].errorbar(gam, mfm, yerr=mf_err, fmt="o-", color="#d62728", capsize=3, label="mf medio")
    l2 = ax2.errorbar(gam, hl, yerr=h_err, fmt="s--", color="#1f77b4", capsize=3, label="convex hull")
    ax[0].set_xlabel(r"fuerza de guiado $\gamma$"); ax[0].set_ylabel("mf medio (EquiNO+$\\sigma^2$)", color="#d62728")
    ax2.set_ylabel("convex hull (diversidad)", color="#1f77b4")
    ax[0].set_xscale("symlog", linthresh=90); ax[0].grid(alpha=0.3)
    ax[0].legend([l1, l2], [l1.get_label(), l2.get_label()], loc="center right")
    ax[1].plot(hl, mfm, "-", color="#888", zorder=1)
    sc = ax[1].scatter(hl, mfm, c=gam, cmap="viridis", s=90, zorder=3)
    for g, h, m in zip(gam, hl, mfm):
        ax[1].annotate(f"$\\gamma$={g}", (h, m), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax[1].set_xlabel("convex hull (diversidad)"); ax[1].set_ylabel("mf medio")
    ax[1].grid(alpha=0.3); fig.colorbar(sc, ax=ax[1], label=r"$\gamma$")
    fig.tight_layout(); fig.savefig(IMG / "fig_res_guidance_strength.png", dpi=160); plt.close()

    with (OUTDIR / "r4b_summary.md").open("w", encoding="utf-8") as fh:
        fh.write(f"# R4b - Fuerza de guiado (equino+sigma2, solid, N={N}, STEPS={STEPS}, bell, seed pareada)\n\n")
        fh.write("gamma=0 y gamma=600 reusados de R3b (mismo protocolo).\n\n")
        fh.write("| gamma | mf medio | IC95 mf | P(mf>0.8) | P(mf>0.9) | hull | IC95 hull |\n|---|---|---|---|---|---|---|\n")
        for g, m, mlo, mhi, p8, p9, h, hlo, hhi in rows:
            fh.write(f"| {g} | {m:.3f} | [{mlo:.3f}, {mhi:.3f}] | {p8:.2f} | {p9:.2f} | {h:.1f} | [{hlo:.1f}, {hhi:.1f}] |\n")
    log(f"\n-> {IMG/'fig_res_guidance_strength.png'}\n-> {OUTDIR/'r4b_summary.md'}\nR4b COMPLETO.")


if __name__ == "__main__":
    main()
