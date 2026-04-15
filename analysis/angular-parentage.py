"""
angular_parentage.py
=====================
MC truth photon angular distributions broken down by parent category.

Mirrors the energy parentage script but with θ and φ on the x-axis.

Physics question:
  Do different photon populations (signal π⁰ daughters, ISR, FSR, other)
  occupy different angular regions? This tells us where in phase space
  our signal actually lives and where contamination enters.

Plots produced per simulation (G4 and CC3):
  1. θ distribution — all categories, raw counts, stacked step histograms
  2. φ distribution — all categories, raw counts, stacked step histograms
  3. θ distribution — genStatus==1 only (same categories)
  4. φ distribution — genStatus==1 only (same categories)

Binning:
  θ: 2° bins, 0–180°   — finer than acceptance scan to show structure
  φ: 3° bins, -180–180° — fine enough to see octagon dips if present

All plots: raw counts, NOT normalized.

Run:
    source ~/source.sh
    python3 angular_parentage.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import edm4hep
import os

# ─── Config ───────────────────────────────────────────────────────────────────
G4_FILE  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-geant4-100kevents-sim.edm4hep.root"
CC3_FILE = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-100kevents-sim.edm4hep.root"

PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/angular_parentage_100k")
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_EVENTS = 100_000
PHOTON_PDG = 22
PI0_PDG    = 111
TAU_PDG    = 15
E_PDG      = 11

# Binning
# θ: 2° — fine enough to see barrel-endcap transition (~40° and ~140°) clearly
# φ: 3° — coarser than θ scan to keep statistics up, still resolves 45° octagon structure
BINS_THETA = np.arange(0,   182, 2)
BINS_PHI   = np.arange(-180, 183, 3)

# ECal barrel octagon boundaries — 8 modules, 45° spacing, starting at -157.5°
PHI_BOUNDARIES = [-180 + 22.5 + 45 * i for i in range(8)]

# Approximate barrel-endcap θ boundaries
THETA_BOUNDARIES = [40.0, 140.0]

# ─── Colors — inherited from approved scheme ──────────────────────────────────
# π⁰ signal = blue (matches G4 color in energy plots — signal is the "truth")
# ISR = red/orange (warm, contamination)
# FSR = green
# Other = purple
C_PI0   = "#2166ac"
C_ISR   = "#d6604d"
C_FSR   = "#4dac26"
C_OTHER = "#7b2d8b"

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


# ─── Loader ───────────────────────────────────────────────────────────────────
def load_file(label, path, max_events=MAX_EVENTS):
    """
    Single pass. For every photon, record:
      - θ, φ (from 4-momentum)
      - energy
      - generatorStatus
      - parent category: pi0 / isr / fsr / other

    Parent category logic (priority order — first match wins):
      pi0:   photon's object index is in the π⁰ daughters set
             (direct daughter only — π⁰ → γγ is a 2-body decay, no chain needed)
      isr:   direct parent PDG == ±11 (electron/positron)
             Note: catches any e± parent, including possible final-state electrons,
             not only beam ISR — but in τ events these are dominated by ISR
      fsr:   direct parent PDG == ±15 (tau)
      other: everything else (secondaries, brem off other particles, etc.)

    Returns dict of numpy arrays, all aligned by index.
    """
    print(f"  Loading {label}...")
    reader = root_io.Reader(path)

    theta_list  = []
    phi_list    = []
    energy_list = []
    genstat_list= []
    cat_list    = []   # 0=pi0, 1=isr, 2=fsr, 3=other

    n_events = 0
    for event in reader.get("events"):
        if n_events >= max_events:
            break
        n_events += 1

        particles = list(event.get("MCParticles"))

        # Build π⁰ daughter index set — one set per event, O(1) lookup below
        pi0_children = set()
        for p in particles:
            if abs(p.getPDG()) == PI0_PDG:
                for child in p.getDaughters():
                    pi0_children.add(child.getObjectID().index)

        for p in particles:
            if p.getPDG() != PHOTON_PDG:
                continue

            p4     = edm4hep.utils.p4(p)
            theta  = np.degrees(float(p4.theta()))
            phi    = np.degrees(float(p4.phi()))
            energy = p.getEnergy()
            genstat= p.getGeneratorStatus()

            # Parent PDGs (direct parents only, one level up)
            parent_pdgs = {abs(par.getPDG()) for par in p.getParents()}
            idx = p.getObjectID().index

            # Category assignment — priority: signal > ISR > FSR > other
            if idx in pi0_children:
                cat = 0
            elif E_PDG in parent_pdgs:
                cat = 1
            elif TAU_PDG in parent_pdgs:
                cat = 2
            else:
                cat = 3

            theta_list.append(theta)
            phi_list.append(phi)
            energy_list.append(energy)
            genstat_list.append(genstat)
            cat_list.append(cat)

    d = {
        "theta" : np.array(theta_list),
        "phi"   : np.array(phi_list),
        "energy": np.array(energy_list),
        "genstat": np.array(genstat_list, dtype=int),
        "cat"   : np.array(cat_list, dtype=int),
    }

    total = len(d["theta"])
    for c, name in enumerate(["pi0", "isr", "fsr", "other"]):
        n = int(np.sum(d["cat"] == c))
        print(f"    {name:6s}: {n:7d}  ({100*n/total:.1f}%)")
    print(f"    Total : {total:7d}  in {n_events} events")

    return d


# ─── Plot helper ──────────────────────────────────────────────────────────────
def plot_angular(angle_arr, cat_arr, bins, xlabel,
                 title, filename,
                 vlines=None, genstat_label=""):
    """
    Single-panel angular distribution, all 4 categories overlaid as step histograms.

    vlines: list of x values for detector boundary markers (vertical dashed lines).
    """
    cats = {
        0: {"color": C_PI0,   "ls": "-",  "label": "Signal (π⁰ daughter)"},
        1: {"color": C_ISR,   "ls": "--", "label": "ISR / e± parent"},
        2: {"color": C_FSR,   "ls": "-.", "label": "FSR (τ parent)"},
        3: {"color": C_OTHER, "ls": ":",  "label": "Other / secondary"},
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    for c, style in cats.items():
        mask = cat_arr == c
        if np.sum(mask) == 0:
            continue
        ax.hist(angle_arr[mask], bins=bins,
                histtype="step", linewidth=2,
                color=style["color"], linestyle=style["ls"],
                label=f"{style['label']} (N={int(np.sum(mask))})")

    # Detector boundary markers
    if vlines is not None:
        ymax = ax.get_ylim()[1]
        for xv in vlines:
            ax.axvline(xv, color="steelblue", linewidth=1.0,
                       linestyle=":", alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Photons / bin  [raw counts]")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(framealpha=0.5, loc="upper right")
    ax.text(0.01, 1.01,
            f"ILD sim — 100k events  |  raw counts{genstat_label}",
            transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

    plt.tight_layout()
    out = f"{PLOT_DIR}/{filename}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {filename}")


# ─── Main ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Angular parentage distributions — 100k events")
print("=" * 60)

print("\n[G4]")
g4 = load_file("Geant4", G4_FILE)

print("\n[CC3]")
cc3 = load_file("CaloClouds3", CC3_FILE)

# ─── G4 plots ─────────────────────────────────────────────────────────────────
print("\n  Plotting G4...")

plot_angular(
    angle_arr = g4["theta"],
    cat_arr   = g4["cat"],
    bins      = BINS_THETA,
    xlabel    = "MC Photon θ [deg]",
    title     = "θ — Photon parentage breakdown | Geant4 sim\nAll MC photons",
    filename  = "g4_theta_all.png",
    vlines    = THETA_BOUNDARIES,
    genstat_label = "  |  all genStatus",
)

plot_angular(
    angle_arr = g4["phi"],
    cat_arr   = g4["cat"],
    bins      = BINS_PHI,
    xlabel    = "MC Photon φ [deg]",
    title     = "φ — Photon parentage breakdown | Geant4 sim\nAll MC photons",
    filename  = "g4_phi_all.png",
    vlines    = PHI_BOUNDARIES,
    genstat_label = "  |  all genStatus",
)

# genStatus == 1 only
gs1_g4 = g4["genstat"] == 1

plot_angular(
    angle_arr = g4["theta"][gs1_g4],
    cat_arr   = g4["cat"][gs1_g4],
    bins      = BINS_THETA,
    xlabel    = "MC Photon θ [deg]",
    title     = "θ — Photon parentage breakdown | Geant4 sim\ngenStatus == 1 only",
    filename  = "g4_theta_gs1.png",
    vlines    = THETA_BOUNDARIES,
    genstat_label = "  |  genStatus==1",
)

plot_angular(
    angle_arr = g4["phi"][gs1_g4],
    cat_arr   = g4["cat"][gs1_g4],
    bins      = BINS_PHI,
    xlabel    = "MC Photon φ [deg]",
    title     = "φ — Photon parentage breakdown | Geant4 sim\ngenStatus == 1 only",
    filename  = "g4_phi_gs1.png",
    vlines    = PHI_BOUNDARIES,
    genstat_label = "  |  genStatus==1",
)

# ─── CC3 plots ────────────────────────────────────────────────────────────────
print("\n  Plotting CC3...")

plot_angular(
    angle_arr = cc3["theta"],
    cat_arr   = cc3["cat"],
    bins      = BINS_THETA,
    xlabel    = "MC Photon θ [deg]",
    title     = "θ — Photon parentage breakdown | CaloClouds3 sim\nAll MC photons",
    filename  = "cc3_theta_all.png",
    vlines    = THETA_BOUNDARIES,
    genstat_label = "  |  all genStatus",
)

plot_angular(
    angle_arr = cc3["phi"],
    cat_arr   = cc3["cat"],
    bins      = BINS_PHI,
    xlabel    = "MC Photon φ [deg]",
    title     = "φ — Photon parentage breakdown | CaloClouds3 sim\nAll MC photons",
    filename  = "cc3_phi_all.png",
    vlines    = PHI_BOUNDARIES,
    genstat_label = "  |  all genStatus",
)

gs1_cc3 = cc3["genstat"] == 1

plot_angular(
    angle_arr = cc3["theta"][gs1_cc3],
    cat_arr   = cc3["cat"][gs1_cc3],
    bins      = BINS_THETA,
    xlabel    = "MC Photon θ [deg]",
    title     = "θ — Photon parentage breakdown | CaloClouds3 sim\ngenStatus == 1 only",
    filename  = "cc3_theta_gs1.png",
    vlines    = THETA_BOUNDARIES,
    genstat_label = "  |  genStatus==1",
)

plot_angular(
    angle_arr = cc3["phi"][gs1_cc3],
    cat_arr   = cc3["cat"][gs1_cc3],
    bins      = BINS_PHI,
    xlabel    = "MC Photon φ [deg]",
    title     = "φ — Photon parentage breakdown | CaloClouds3 sim\ngenStatus == 1 only",
    filename  = "cc3_phi_gs1.png",
    vlines    = PHI_BOUNDARIES,
    genstat_label = "  |  genStatus==1",
)

print(f"\n  8 plots saved to: {PLOT_DIR}")
print("  G4:  g4_theta_all, g4_phi_all, g4_theta_gs1, g4_phi_gs1")
print("  CC3: cc3_theta_all, cc3_phi_all, cc3_theta_gs1, cc3_phi_gs1\n")
