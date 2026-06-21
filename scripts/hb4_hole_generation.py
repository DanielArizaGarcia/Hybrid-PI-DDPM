"""HB4 - Prueba de oro: GENERACION en HOLE a 1000 pasos con guiado ENMASCARADO.

Architect hole (frozen, 1000 pasos DDPM fieles) genera figuras con hueco. Se comparan 4 arms:
  no_guide | pb_bell (PB-PUNet+campana) | eq_bell (equino+sigma2+campana) | eq_gate (equino+sigma2 gate por sigma2)

CLAVE METODOLOGICA: el guiado optimiza el mf ENMASCARADO (solo material). La mascara se deriva del
estimado limpio x0 (Tweedie) en cada paso -> el guiado NO puede subir el mf rellenando el hueco
(artefacto que un guiado sin mascara provocaria). Diagnostico hole_frac confirma que el hueco se conserva.

Metricas: mf-mean enmascarado (juez doble: equino+sigma2 [circular] y pb_unet [familia distinta]),
P(mf>0.9), hole_frac, diversidad (hull/Vendi via VAE-hole inline + pixel). Bootstrap IC95%.

Colab (Drive montado + dataset en /content/Datos/processed_image2):
  !cd <repo> && git pull && uv run --with scipy python scripts/hb4_hole_generation.py
Local:
  uv run --with scipy python scripts/hb4_hole_generation.py
"""
from __future__ import annotations
import sys, os, math, time, glob
os.environ["MPLBACKEND"] = "Agg"
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
else:
    ROOT = Path(r"G:\Mi unidad\Hybrid-PI-DDPM-runtime")
    DATA = Path(r"C:\Datos\processed_image2")
OUT = ROOT / "artifacts" / "hb4_hole_generation"
MENG = ROOT / "models" / "engineer"
MARCH = ROOT / "models" / "architect"
ARCH = MARCH / "20260621_103827_colab_e6_architect_hole" / "best.pt"
PBUNET = MENG / "20260621_105401_colab_e6_pb_unet_hole_masked" / "best.pt"
EQS2 = MENG / "20260621_123532_colab_e6_equino_unc_hole_masked" / "best.pt"
VAE_CK = OUT / "vae_hole.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = int(os.environ.get("HB4_N", 256)); BATCH = 8
STEPS = int(os.environ.get("HB4_STEPS", 1000)); SCALE = 600.0
W_MAX, BELL_PEAK, BELL_WIDTH, GRAD_CLIP, SEED = 1.0, 0.5, 0.22, 20.0, 7
TAU = 1.63                # recalibracion sigma2 (de R2)
MASK_EPS = 0.02           # material = z_real > MASK_EPS * z_max (verificado: hueco => z==0)
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


# ---------------- VAE (entrena inline sobre HOLE si no esta) ----------------
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
        vae.load_state_dict(torch.load(VAE_CK, map_location=DEVICE)); log("VAE-hole cargada de", VAE_CK); return vae.eval()
    log("Entrenando VAE inline sobre HOLE...")
    files = sorted(f for f in glob.glob(str(DATA / "*.npz")) if "hole" in os.path.basename(f))
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
    log("VAE-hole entrenada y guardada en", VAE_CK); return vae.eval()


# ---------------- generacion (guiado ENMASCARADO anti-relleno) ----------------
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
                abar = scheduler.alphas_cumprod[t].to(DEVICE)
            if arm != "no_guide":
                xr = x.detach().clone().requires_grad_(True)
                xr_e = renorm(xr, a_min, a_max, e_min, e_max)
                out = engineer(torch.cat([xr_e, fzc], dim=1), tb)
                mf_map = compute_membrane_factor_from_prediction(out.sample, pmb, psb)  # (bs,1,64,64)
                with torch.no_grad():
                    x0 = torch.sqrt(abar) * xr.detach() - torch.sqrt(1.0 - abar) * v   # Tweedie x0 (v-pred)
                    x0r = ((x0 + 1) / 2) * (a_max - a_min) + a_min
                    mask = (x0r > MASK_EPS * a_max).float()                              # material
                    denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
                masked_mf = (mf_map * mask).sum(dim=(1, 2, 3)) / denom                   # mf SOLO material
                g = torch.autograd.grad(((1.0 - masked_mf) ** 2).sum(), xr)[0]
                env = SCALE * W_MAX * bell_env(i, total)
                if arm == "eq_gate":
                    with torch.no_grad():
                        sig2 = TAU * torch.exp(out.log_variance[:, :1])
                        inv = 1.0 / (sig2 + 1e-6)
                        gate = torch.clamp(inv / inv.mean(dim=(2, 3), keepdim=True), 0.2, 5.0)
                    grad = torch.clamp(env * gate * g, -GRAD_CLIP, GRAD_CLIP)
                else:
                    grad = torch.clamp(env * g, -GRAD_CLIP, GRAD_CLIP)
                v = v + torch.sqrt(1.0 - abar) * grad
                del out, xr, mf_map, g
            x = scheduler.step(v, t, x).prev_sample
        zs.append(x.detach())
    z = torch.cat(zs, 0)
    zr = ((z.cpu().numpy() + 1) / 2) * (a_max - a_min) + a_min
    return zr[:, 0], z   # (n,64,64) geometria real, y z normalizado (n,1,64,64)


def judge_mf_masked(z_norm_arch, a_stats, judge, j_stats):
    """mf-mean ENMASCARADO (material) a t=0, juzgado por 'judge'. Mascara de la geometria generada."""
    a_min, a_max = float(a_stats["z_min"]), float(a_stats["z_max"])
    j_min, j_max = float(j_stats["z_min"]), float(j_stats["z_max"])
    pm = torch.tensor(j_stats["physics_mean"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    ps = torch.tensor(j_stats["physics_std"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mfs = []
    with torch.no_grad():
        for s in range(0, z_norm_arch.shape[0], BATCH):
            zb = z_norm_arch[s:s + BATCH].to(DEVICE)
            ze = renorm(zb, a_min, a_max, j_min, j_max)
            zr = ((zb + 1) / 2) * (a_max - a_min) + a_min
            mask = (zr > MASK_EPS * a_max).float()
            denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            fz0 = torch.zeros((zb.shape[0], 1, 64, 64), device=DEVICE)
            tb0 = torch.zeros(zb.shape[0], device=DEVICE, dtype=torch.long)
            pred = judge(torch.cat([ze, fz0], dim=1), tb0).sample
            mf_map = compute_membrane_factor_from_prediction(pred, pm.expand(zb.shape[0], -1, -1, -1),
                                                             ps.expand(zb.shape[0], -1, -1, -1))
            mf = (mf_map * mask).sum(dim=(1, 2, 3)) / denom
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
def hole_frac(Z):  # fraccion de pixeles de hueco por muestra (z ~ 0)
    return (Z <= MASK_EPS * Z.max()).reshape(len(Z), -1).mean(axis=1)


def grid(Z, path, title):
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for k in range(8):
        a = ax[k // 4, k % 4]; a.imshow(Z[k], cmap="viridis"); a.set_title(f"#{k}", fontsize=8); a.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"device={DEVICE} N={N} STEPS={STEPS} | OUT={OUT}")
    for p in (ARCH, PBUNET, EQS2):
        if not p.exists(): log("FALTA:", p); return
    architect, ack = load(ARCH); a_stats = ack["normalization_stats"]
    pb, pbk = load(PBUNET); eq, eqk = load(EQS2)
    scheduler = build_scheduler(ack["model_config"]); scheduler.set_timesteps(STEPS, device=DEVICE)
    zmin, zmax = float(a_stats["z_min"]), float(a_stats["z_max"])
    vae = get_vae(zmin, zmax)

    ARMS = [("no_guide", None, None), ("pb_bell", pb, pbk["normalization_stats"]),
            ("eq_bell", eq, eqk["normalization_stats"]), ("eq_gate", eq, eqk["normalization_stats"])]
    geoms, znorm = {}, {}
    for arm, eng, est in ARMS:
        t0 = time.time(); log(f"  generando {arm}...")
        zr, zn = gen(arm, architect, a_stats, eng, est, scheduler, N, SEED)
        geoms[arm] = zr; znorm[arm] = zn
        grid(zr, OUT / f"grid_{arm}.png", f"HB4 {arm} (hole, STEPS={STEPS})")
        np.savez(OUT / "hb4_geoms.npz", **{f"Z_{a}": geoms[a] for a in geoms})  # incremental: salva tras cada arm
        log(f"  generado {arm} ({time.time()-t0:.0f}s)  hole_frac={hole_frac(zr).mean():.3f}  [guardado]")
    log("geometrias guardadas:", OUT / "hb4_geoms.npz")

    # juez doble (enmascarado): equino+s2 (circular) y pb_unet (familia distinta)
    mf_eq = {a: judge_mf_masked(znorm[a], a_stats, eq, eqk["normalization_stats"]) for a in geoms}
    mf_pb = {a: judge_mf_masked(znorm[a], a_stats, pb, pbk["normalization_stats"]) for a in geoms}
    hfr = {a: hole_frac(geoms[a]) for a in geoms}
    lat = {}
    with torch.no_grad():
        for a in geoms:
            zn = 2 * (geoms[a] - zmin) / (zmax - zmin) - 1
            mu, _ = vae.encode(torch.from_numpy(zn).unsqueeze(1).float().to(DEVICE)); lat[a] = mu.cpu().numpy()

    rows = {}
    log(f"\n{'arm':9s} | mf_eq  mf_pb  P90eq | hole% | hull  Vlat  Vpx")
    for a in geoms:
        rows[a] = dict(mf_eq=float(mf_eq[a].mean()), mf_pb=float(mf_pb[a].mean()),
                       p90=float((mf_eq[a] > 0.9).mean()), hole=float(hfr[a].mean()),
                       hull=hull_area(lat[a]), vlat=vendi(lat[a]), vpx=vendi(geoms[a].reshape(N, -1)))
        r = rows[a]
        log(f"{a:9s} | {r['mf_eq']:.3f}  {r['mf_pb']:.3f}  {r['p90']:.2f} | {r['hole']*100:4.1f}% | {r['hull']:5.2f} {r['vlat']:4.2f} {r['vpx']:4.2f}")

    # bootstrap: cada arm guiado vs no_guide
    rng = np.random.default_rng(0); METR = ["mf_eq", "mf_pb", "hole", "hull"]
    def mval(a, idx, m):
        if m == "mf_eq": return float(mf_eq[a][idx].mean())
        if m == "mf_pb": return float(mf_pb[a][idx].mean())
        if m == "hole": return float(hfr[a][idx].mean())
        if m == "hull": return hull_area(lat[a][idx])
    boot = {a: {m: np.empty(B_BOOT) for m in METR} for a in geoms}
    log(f"\nbootstrap B={B_BOOT}...")
    for a in geoms:
        for b in range(B_BOOT):
            idx = rng.integers(0, N, N)
            for m in METR: boot[a][m][b] = mval(a, idx, m)

    names = {"mf_eq": "mf(eq+s2)", "mf_pb": "mf(pb)", "hole": "hole_frac", "hull": "hull"}
    with (OUT / "hb4_summary.md").open("w", encoding="utf-8") as fh:
        fh.write(f"# HB4 - Generacion HOLE 1000 pasos, guiado enmascarado (N={N})\n\n")
        fh.write("Referencia datos reales: mf_true enmascarado ~0.75.\n\n")
        fh.write("| arm | mf(eq+s2) | mf(pb) | P(mf>0.9) | hole_frac | hull | Vendi_lat | Vendi_px |\n|---|---|---|---|---|---|---|---|\n")
        for a in geoms:
            r = rows[a]
            fh.write(f"| {a} | {r['mf_eq']:.3f} | {r['mf_pb']:.3f} | {r['p90']:.2f} | {r['hole']*100:.1f}% | {r['hull']:.2f} | {r['vlat']:.2f} | {r['vpx']:.2f} |\n")
        fh.write(f"\n## Bootstrap (B={B_BOOT}, IC95%, * = excluye 0) vs no_guide\n\n")
        fh.write("| comparacion | metrica | dif | IC95% | P(A>B) |\n|---|---|---|---|---|\n")
        for A in ("pb_bell", "eq_bell", "eq_gate"):
            for m in METR:
                d = boot[A][m] - boot["no_guide"][m]; lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
                sig = "*" if (lo > 0 or hi < 0) else ""
                fh.write(f"| {A} vs no_guide | {names[m]} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}]{sig} | {(d>0).mean():.3f} |\n")
                log(f"  {A} vs no_guide {names[m]:10s} dif={d.mean():+.3f} IC[{lo:+.3f},{hi:+.3f}]{sig} P={float((d>0).mean()):.3f}")
        # eq_gate vs pb_bell (nuestra propuesta vs baseline guiado)
        fh.write("\n| comparacion | metrica | dif | IC95% | P(A>B) |\n|---|---|---|---|---|\n")
        for m in METR:
            d = boot["eq_gate"][m] - boot["pb_bell"][m]; lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            sig = "*" if (lo > 0 or hi < 0) else ""
            fh.write(f"| eq_gate vs pb_bell | {names[m]} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}]{sig} | {(d>0).mean():.3f} |\n")

    plt.figure(figsize=(7.5, 6)); col = {"no_guide": "#7f7f7f", "pb_bell": "#000", "eq_bell": "#1f77b4", "eq_gate": "#d62728"}
    for a in geoms:
        plt.scatter(rows[a]["hull"], rows[a]["mf_eq"], s=120, color=col[a], zorder=3)
        plt.annotate(a, (rows[a]["hull"], rows[a]["mf_eq"]), textcoords="offset points", xytext=(8, 4))
    plt.axhline(0.75, ls="--", color="green", alpha=0.6, label="mf datos reales (~0.75)")
    plt.xlabel("diversidad: hull area (VAE-hole)"); plt.ylabel("calidad: mf enmascarado (juez eq+s2)")
    plt.title(f"HB4 HOLE (N={N}, STEPS={STEPS}) - frontera calidad/diversidad"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_hb4_pareto.png", dpi=160); plt.close()
    log(f"\nTODO guardado en {OUT}\nHB4 COMPLETO.")


if __name__ == "__main__":
    main()
