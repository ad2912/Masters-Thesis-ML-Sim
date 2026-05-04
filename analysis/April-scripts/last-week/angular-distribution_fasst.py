import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ─── Config ───────────────────────────────────────────────────────────────────
NPZ_DIR = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/analysis/npz_files"
GEN_NPZ = f"{NPZ_DIR}/gen_master.npz"
G4_NPZ  = f"{NPZ_DIR}/g4_master.npz"
CC3_NPZ = f"{NPZ_DIR}/cc3_master.npz"

PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/angular_distributions_fasst_100k")
os.makedirs(PLOT_DIR, exist_ok=True)

E_THRESH = 10.0

# ─── Colors ───────────────────────────────────────────────────────────────────
C_GEN    = "#333333"   # dark gray
C_G4     = "#2166ac"   # blue
C_CC3    = "#d6604d"   # red
C_CC_FS  = "#f4a582"   # orange
C_CC_G4  = "#4dac26"   # green
C_MISSED = "#7b2d8b"   # purple

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 12,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 12,
    "legend.fontsize"  : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})

BINS_THETA = np.linspace(0,   180, 37)
BINS_PHI   = np.linspace(-180, 180, 73)

# Boundaries
THETA_VLINES = [10.0, 35.0, 40.0, 140.0, 145.0, 170.0]
# New logic: Start at -157.5, then every 45 degrees
PHI_VLINES   = [-157.5 + 45 * i for i in range(8)]

# ─── Plot function ────────────────────────────────────────────────────────────
def plot_hist(angle_type, bins, title, filename,
              gen_vals=None, g4_vals=None,
              cc3_total_vals=None,
              cc3_fs_vals=None, cc3_g4_vals=None,
              cc3_missed_vals=None):
    
    fig, ax = plt.subplots(figsize=(10, 5))

    def _hist(vals, color, label, ls="-", lw=2):
        if vals is not None and len(vals) > 0:
            ax.hist(vals, bins=bins, histtype="step",
                    color=color, linewidth=lw, linestyle=ls, label=label)

    _hist(gen_vals,  C_GEN,  f"GEN (N={len(gen_vals) if gen_vals is not None else 0})")
    _hist(g4_vals,   C_G4,   f"G4 (N={len(g4_vals) if g4_vals is not None else 0})")
    _hist(cc3_total_vals, C_CC3, f"CC3 total (N={len(cc3_total_vals) if cc3_total_vals is not None else 0})", ls="--")
    _hist(cc3_fs_vals,  C_CC_FS, f"CC3 fast-sim (N={len(cc3_fs_vals) if cc3_fs_vals is not None else 0})", ls="--")
    _hist(cc3_g4_vals,  C_CC_G4, f"CC3 G4-handled (N={len(cc3_g4_vals) if cc3_g4_vals is not None else 0})", ls="-.")
    _hist(cc3_missed_vals, C_MISSED, f"CC3 missed by fast-sim (N={len(cc3_missed_vals) if cc3_missed_vals is not None else 0})")

    # Detector markers and X-labels
    if angle_type == "theta":
        for xv in THETA_VLINES: ax.axvline(xv, color="gray", lw=0.8, ls=":", alpha=0.7)
        ax.set_xlabel("MC photon θ [deg]")
    else:
        for xv in PHI_VLINES: ax.axvline(xv, color="steelblue", lw=1.0, ls=":", alpha=0.6)
        ax.set_xlabel("MC photon φ [deg]")

    ax.set_ylabel("Photons / bin [raw counts]")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(framealpha=0.5, loc="upper right")
    ax.text(0.01, 1.01, "ILD sim — 100k events | raw counts | preliminary",
            transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/{filename}", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {filename}")

# ─── Main ─────────────────────────────────────────────────────────────────────
print("\nLoading NPZ files...")
gen = np.load(GEN_NPZ)
g4  = np.load(G4_NPZ)
cc3 = np.load(CC3_NPZ)

for angle in ("theta", "phi"):
    v = angle
    b = BINS_THETA if angle == "theta" else BINS_PHI

    # Selection masks
    # 1. All
    plot_hist(angle, b, f"{angle.upper()} — All Photons", f"1_all_{angle}.png",
              gen_vals=gen[v], g4_vals=g4[v], cc3_total_vals=cc3[v])

    # 2. genStatus == 1
    m_gen2, m_g42, m_cc2 = gen['stat']==1, g4['stat']==1, cc3['stat']==1
    plot_hist(angle, b, f"{angle.upper()} — genStatus==1", f"2_gs1_{angle}.png",
              gen_vals=gen[v][m_gen2], g4_vals=g4[v][m_g42], cc3_total_vals=cc3[v][m_cc2])

    # 3. E >= 10 GeV
    m_gen3, m_g43, m_cc3 = (gen['stat']==1)&(gen['e']>=E_THRESH), (g4['stat']==1)&(g4['e']>=E_THRESH), (cc3['stat']==1)&(cc3['e']>=E_THRESH)
    plot_hist(angle, b, f"{angle.upper()} — genStatus==1, E≥10 GeV", f"3_gs1_e10_{angle}.png",
              gen_vals=gen[v][m_gen3], g4_vals=g4[v][m_g43], 
              cc3_fs_vals=cc3[v][m_cc3 & cc3['fs']], cc3_g4_vals=cc3[v][m_cc3 & ~cc3['fs']])

    # 4. Pi0 Daughters
    m_gen4, m_g44, m_cc4 = m_gen3 & (gen['cat']==0), m_g43 & (g4['cat']==0), m_cc3 & (cc3['cat']==0)
    plot_hist(angle, b, f"{angle.upper()} — GS1, E≥10, π⁰ Daughter", f"4_gs1_e10_pi0_{angle}.png",
              gen_vals=gen[v][m_gen4], g4_vals=g4[v][m_g44],
              cc3_fs_vals=cc3[v][m_cc4 & cc3['fs']], cc3_g4_vals=cc3[v][m_cc4 & ~cc3['fs']])

    # 5. Not ISR
    m_gen5, m_g45, m_cc5 = m_gen3 & (gen['cat']!=1), m_g43 & (g4['cat']!=1), m_cc3 & (cc3['cat']!=1)
    plot_hist(angle, b, f"{angle.upper()} — GS1, E≥10, Not ISR", f"5_gs1_e10_notisr_{angle}.png",
              gen_vals=gen[v][m_gen5], g4_vals=g4[v][m_g45],
              cc3_fs_vals=cc3[v][m_cc5 & cc3['fs']], cc3_g4_vals=cc3[v][m_cc5 & ~cc3['fs']])

    # 6. Missed by Fast Sim (CC3 only)
    m_cc6 = m_cc5 & ~cc3['fs']
    plot_hist(angle, b, f"{angle.upper()} — GS1, E≥10, Not ISR, G4-handled", f"6_missed_{angle}.png",
              cc3_missed_vals=cc3[v][m_cc6])

print(f"\nDone! 12 plots in {PLOT_DIR}")
