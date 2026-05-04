"""
angular_distributions.py
=========================
MC truth photon angular distributions (θ and φ).
Compares GEN input, Geant4 sim, and CaloClouds3 sim.

Plots produced (θ and φ for each = 12 figures total):
  1. All photons                          — GEN, G4, CC3 total
  2. genStatus == 1                       — GEN, G4, CC3 total
  3. genStatus==1, E>=10 GeV             — GEN, G4, CC3 split (FS + G4-handled)
  4. genStatus==1, E>=10 GeV, pi0 daughter — GEN, G4, CC3 split
  5. genStatus==1, E>=10 GeV, not ISR    — GEN, G4, CC3 split
  6. genStatus==1, E>=10 GeV, not ISR,
     NOT handled by fast sim (CC3 only)  — CC3 only, no G4/GEN overlay

All histograms: raw counts, NOT normalized.
GEN file capped at 100k events to match sim statistics.

Run:
    source ~/source.sh
    python3 angular_distributions.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import edm4hep
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
GEN_FILE = "/data/dust/user/alimuham/thesis/InputFiles/tau_pi0_10GeV_filtered_500kevents.edm4hep.root"
G4_FILE  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-geant4-100kevents-sim.edm4hep.root"
CC3_FILE = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-100kevents-sim.edm4hep.root"

PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/angular_distributions_100k")
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_EVENTS  = 100_000
E_THRESH    = 10.0
PHOTON_PDG  = 22
PI0_PDG     = 111

# ─── Colors ───────────────────────────────────────────────────────────────────
C_GEN    = "#333333"   # dark gray  — generator
C_G4     = "#2166ac"   # blue       — Geant4
C_CC3    = "#d6604d"   # red        — CC3 total
C_CC_FS  = "#f4a582"   # orange     — CC3 fast-sim handled
C_CC_G4  = "#4dac26"   # green      — CC3 G4-handled
C_MISSED = "#7b2d8b"   # purple     — missed by fast sim

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

# ─── Binning ──────────────────────────────────────────────────────────────────
# θ: 5° bins across 0–180°
# φ: 5° bins across -180–180°
BINS_THETA = np.linspace(0,   180, 37)   # 36 bins × 5°
BINS_PHI   = np.linspace(-180, 180, 73)  # 72 bins × 5°

# ─── Detector markers ─────────────────────────────────────────────────────────
# θ: barrel-endcap transition boundaries (from previous acceptance scan)
THETA_VLINES = [10.0, 35.0, 40.0, 140.0, 145.0, 170.0]

# φ: ILD ECal barrel is octagonal — 8 boundaries every 45°
# Exact offset unknown until we see the data; mark every 45° starting at -180°
PHI_VLINES = [-180 + 45 * i for i in range(9)]   # -180, -135, ..., 180


# ─── Loader ───────────────────────────────────────────────────────────────────
def load_file(label, path, is_cc3=False, max_events=MAX_EVENTS):
    print(f"  Loading {label} ...")
    reader = root_io.Reader(path)

    theta_arr   = []
    phi_arr     = []
    energy_arr  = []
    genstat_arr = []
    is_pi0_arr  = []
    is_isr_arr  = []
    is_fs_arr   = []

    n_events = 0
    for event in reader.get("events"):
        if n_events >= max_events:
            break
        n_events += 1

        particles = list(event.get("MCParticles"))

        # Build pi0-daughter index set once per event — O(n_particles)
        pi0_children = set()
        for p in particles:
            if abs(p.getPDG()) == PI0_PDG:
                for child in p.getDaughters():
                    pi0_children.add(child.getObjectID().index)

        for p in particles:
            if p.getPDG() != PHOTON_PDG:
                continue

            p4    = edm4hep.utils.p4(p)
            theta = np.degrees(float(p4.theta()))
            phi   = np.degrees(float(p4.phi()))

            parent_pdgs = {abs(par.getPDG()) for par in p.getParents()}
            is_isr      = 11 in parent_pdgs
            idx         = p.getObjectID().index

            theta_arr.append(theta)
            phi_arr.append(phi)
            energy_arr.append(p.getEnergy())
            genstat_arr.append(p.getGeneratorStatus())
            is_pi0_arr.append(idx in pi0_children)
            is_isr_arr.append(is_isr)
            is_fs_arr.append(bool(p.isHandledByFastSim()) if is_cc3 else False)

    d = {
        "theta"  : np.array(theta_arr),
        "phi"    : np.array(phi_arr),
        "energy" : np.array(energy_arr),
        "genstat": np.array(genstat_arr, dtype=int),
        "is_pi0" : np.array(is_pi0_arr,  dtype=bool),
        "is_isr" : np.array(is_isr_arr,  dtype=bool),
        "is_fs"  : np.array(is_fs_arr,   dtype=bool),
    }
    n_ph = len(d["theta"])
    n_fs = int(np.sum(d["is_fs"])) if is_cc3 else 0
    print(f"    {n_events} events | {n_ph} photons"
          + (f" | {n_fs} fast-sim" if is_cc3 else ""))
    return d, n_events


# ─── Plot function ────────────────────────────────────────────────────────────
def plot_hist(angle, bins, title, filename,
              gen_vals=None, g4_vals=None,
              cc3_total_vals=None,
              cc3_fs_vals=None, cc3_g4_vals=None,
              cc3_missed_vals=None,
              n_gen=None, n_g4=None, n_cc3=None):
    """
    Flexible histogram plotter.
    Pass the arrays you want drawn; pass None to skip.
    split_cc mode: pass cc3_fs_vals and cc3_g4_vals instead of cc3_total_vals.
    missed mode:   pass only cc3_missed_vals.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    def _hist(vals, color, label, ls="-", lw=2):
        ax.hist(vals, bins=bins, histtype="step",
                color=color, linewidth=lw, linestyle=ls, label=label)

    if gen_vals is not None:
        _hist(gen_vals,  C_GEN,  f"GEN (N={len(gen_vals)}, {n_gen}ev)")
    if g4_vals is not None:
        _hist(g4_vals,   C_G4,   f"G4 (N={len(g4_vals)}, {n_g4}ev)")
    if cc3_total_vals is not None:
        _hist(cc3_total_vals, C_CC3, f"CC3 total (N={len(cc3_total_vals)}, {n_cc3}ev)", ls="--")
    if cc3_fs_vals is not None:
        _hist(cc3_fs_vals,  C_CC_FS, f"CC3 fast-sim (N={len(cc3_fs_vals)})", ls="--")
    if cc3_g4_vals is not None:
        _hist(cc3_g4_vals,  C_CC_G4, f"CC3 G4-handled (N={len(cc3_g4_vals)})", ls="-.")
    if cc3_missed_vals is not None:
        _hist(cc3_missed_vals, C_MISSED,
              f"CC3 missed by fast-sim (N={len(cc3_missed_vals)}, {n_cc3}ev)")

    # Detector boundary markers
    if angle == "theta":
        for xv in THETA_VLINES:
            ax.axvline(xv, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.axvline(THETA_VLINES[0], color="gray", linewidth=0.8,
                   linestyle=":", alpha=0.7, label="detector boundaries")
        ax.set_xlabel("MC photon θ [deg]")
    else:
        for xv in PHI_VLINES:
            ax.axvline(xv, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.axvline(PHI_VLINES[0], color="gray", linewidth=0.8,
                   linestyle=":", alpha=0.7, label="ECal octagon boundaries (45°)")
        ax.set_xlabel("MC photon φ [deg]")

    ax.set_ylabel("Photons / bin  [raw counts]")
    ax.set_title(title)
    ax.legend(framealpha=0.5, fontsize=9)
    ax.text(0.01, 1.01, "ILD sim — 100k events  |  raw counts, not normalized  |  preliminary",
            transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

    plt.tight_layout()
    out = f"{PLOT_DIR}/{filename}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {filename}")


# ─── Main ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Angular distributions — 100k events")
print("=" * 60 + "\n")

gen, n_gen = load_file("GEN (input)",    GEN_FILE, is_cc3=False)
g4,  n_g4  = load_file("G4  (full sim)", G4_FILE,  is_cc3=False)
cc3, n_cc3 = load_file("CC3 (fast sim)", CC3_FILE, is_cc3=True)

print()

# ─── Selection masks ──────────────────────────────────────────────────────────
def sel(d, genstat=None, emin=None, pi0=False, not_isr=False, missed=False):
    m = np.ones(len(d["theta"]), dtype=bool)
    if genstat is not None : m &= d["genstat"] == genstat
    if emin    is not None : m &= d["energy"]  >= emin
    if pi0                 : m &= d["is_pi0"]
    if not_isr             : m &= ~d["is_isr"]
    if missed              : m &= ~d["is_fs"]
    return m

# ─── Generate all plots ───────────────────────────────────────────────────────
for angle in ("theta", "phi"):
    v = angle  # shorthand for array key

    # ── Plot 1: all photons ───────────────────────────────────────────────────
    plot_hist(
        angle, BINS_THETA if angle == "theta" else BINS_PHI,
        title    = f"{angle.upper()} — All photons | GEN vs G4 vs CC3",
        filename = f"1_all_{angle}.png",
        gen_vals = gen[v],
        g4_vals  = g4[v],
        cc3_total_vals = cc3[v],
        n_gen=n_gen, n_g4=n_g4, n_cc3=n_cc3,
    )

    # ── Plot 2: genStatus == 1 ────────────────────────────────────────────────
    m_gen_gs1 = sel(gen, genstat=1)
    m_g4_gs1  = sel(g4,  genstat=1)
    m_cc3_gs1 = sel(cc3, genstat=1)

    plot_hist(
        angle, BINS_THETA if angle == "theta" else BINS_PHI,
        title    = f"{angle.upper()} — genStatus==1 | GEN vs G4 vs CC3",
        filename = f"2_gs1_{angle}.png",
        gen_vals = gen[v][m_gen_gs1],
        g4_vals  = g4[v][m_g4_gs1],
        cc3_total_vals = cc3[v][m_cc3_gs1],
        n_gen=n_gen, n_g4=n_g4, n_cc3=n_cc3,
    )

    # ── Plot 3: genStatus==1, E>=10 GeV ──────────────────────────────────────
    m_gen_3 = sel(gen, genstat=1, emin=E_THRESH)
    m_g4_3  = sel(g4,  genstat=1, emin=E_THRESH)
    m_cc3_3 = sel(cc3, genstat=1, emin=E_THRESH)
    m_cc3_3_fs  = m_cc3_3 &  cc3["is_fs"]
    m_cc3_3_g4  = m_cc3_3 & ~cc3["is_fs"]

    plot_hist(
        angle, BINS_THETA if angle == "theta" else BINS_PHI,
        title    = f"{angle.upper()} — genStatus==1, E≥10 GeV | GEN vs G4 vs CC3 split",
        filename = f"3_gs1_e10_{angle}.png",
        gen_vals    = gen[v][m_gen_3],
        g4_vals     = g4[v][m_g4_3],
        cc3_fs_vals = cc3[v][m_cc3_3_fs],
        cc3_g4_vals = cc3[v][m_cc3_3_g4],
        n_gen=n_gen, n_g4=n_g4, n_cc3=n_cc3,
    )

    # ── Plot 4: genStatus==1, E>=10 GeV, pi0 daughter ────────────────────────
    m_gen_4 = sel(gen, genstat=1, emin=E_THRESH, pi0=True)
    m_g4_4  = sel(g4,  genstat=1, emin=E_THRESH, pi0=True)
    m_cc3_4 = sel(cc3, genstat=1, emin=E_THRESH, pi0=True)
    m_cc3_4_fs  = m_cc3_4 &  cc3["is_fs"]
    m_cc3_4_g4  = m_cc3_4 & ~cc3["is_fs"]

    plot_hist(
        angle, BINS_THETA if angle == "theta" else BINS_PHI,
        title    = f"{angle.upper()} — genStatus==1, E≥10 GeV, π⁰ daughter | GEN vs G4 vs CC3 split",
        filename = f"4_gs1_e10_pi0_{angle}.png",
        gen_vals    = gen[v][m_gen_4],
        g4_vals     = g4[v][m_g4_4],
        cc3_fs_vals = cc3[v][m_cc3_4_fs],
        cc3_g4_vals = cc3[v][m_cc3_4_g4],
        n_gen=n_gen, n_g4=n_g4, n_cc3=n_cc3,
    )

    # ── Plot 5: genStatus==1, E>=10 GeV, not ISR ─────────────────────────────
    m_gen_5 = sel(gen, genstat=1, emin=E_THRESH, not_isr=True)
    m_g4_5  = sel(g4,  genstat=1, emin=E_THRESH, not_isr=True)
    m_cc3_5 = sel(cc3, genstat=1, emin=E_THRESH, not_isr=True)
    m_cc3_5_fs  = m_cc3_5 &  cc3["is_fs"]
    m_cc3_5_g4  = m_cc3_5 & ~cc3["is_fs"]

    plot_hist(
        angle, BINS_THETA if angle == "theta" else BINS_PHI,
        title    = f"{angle.upper()} — genStatus==1, E≥10 GeV, not ISR | GEN vs G4 vs CC3 split",
        filename = f"5_gs1_e10_notisr_{angle}.png",
        gen_vals    = gen[v][m_gen_5],
        g4_vals     = g4[v][m_g4_5],
        cc3_fs_vals = cc3[v][m_cc3_5_fs],
        cc3_g4_vals = cc3[v][m_cc3_5_g4],
        n_gen=n_gen, n_g4=n_g4, n_cc3=n_cc3,
    )

    # ── Plot 6: missed by fast sim (CC3 only) ─────────────────────────────────
    m_cc3_missed = sel(cc3, genstat=1, emin=E_THRESH, not_isr=True, missed=True)

    plot_hist(
        angle, BINS_THETA if angle == "theta" else BINS_PHI,
        title    = f"{angle.upper()} — genStatus==1, E≥10 GeV, not ISR, NOT fast-simmed | CC3 only",
        filename = f"6_missed_{angle}.png",
        cc3_missed_vals = cc3[v][m_cc3_missed],
        n_cc3=n_cc3,
    )

print(f"\n  All plots saved to: {PLOT_DIR}")
print(f"  12 figures total (6 selections × θ + φ)\n")
