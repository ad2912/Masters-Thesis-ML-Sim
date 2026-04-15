"""
angular_distributions_sim.py
=============================
Sim-level MC truth photon angular distributions.
Compares: Generator (1k events) vs Geant4 sim vs CaloClouds3 sim.

Produces θ and φ histograms for 6 photon selections × 3 detector regions = 36 plots.
For selections involving genStat==1 and/or E>=10 GeV, CC3 is split into:
  - CC3 fast sim  (isHandledByFastSim == True)
  - CC3 Geant4    (isHandledByFastSim == False)

Output: ~/thesis-ml-sim/plots/angular_sim_V2/

Run with:
    source ~/source.sh
    python3 angular_distributions_sim.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import edm4hep
import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
GEN_FILE = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/steering/tau_pi0_10GeV_filtered.edm4hep.root"
G4_FILE  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-geant4-V2-sim.edm4hep.root"
CC_FILE  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-V2-sim.edm4hep.root"

PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/angular_sim_V2")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PHOTON_PDG   = 22
PI0_PDG      = 111
E_THRESHOLD  = 10.0   # GeV — CaloClouds3 trigger threshold
MAX_GEN_EVENTS = 1000

# Detector regions in degrees (based on previous plots)
BARREL_MIN, BARREL_MAX   = 45.0, 130.0
ENDCAP_MAX_LOW           = 25.0          # 0–25°
ENDCAP_MIN_HIGH          = 150.0         # 150–180°

# ─────────────────────────────────────────────────────────────────────────────
# COLORS  (consistent with previous analyses your supervisor has seen)
# ─────────────────────────────────────────────────────────────────────────────
C_GEN    = "#333333"   # dark gray  — generator truth
C_G4     = "#2166ac"   # blue       — Geant4
C_CC     = "#d6604d"   # red        — CC3 total
C_CC_FS  = "#f4a582"   # orange     — CC3 fast sim (isHandledByFastSim)
C_CC_G4  = "#4dac26"   # green      — CC3 Geant4-handled

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 12,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 12,
    "legend.fontsize"  : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# PARENTAGE CRAWL
# ─────────────────────────────────────────────────────────────────────────────
def has_pi0_ancestor(particle):
    """
    Walk up the parent chain. Return True if any ancestor has PDG == 111 (π⁰).
    Visited set prevents infinite loops from circular generator records.
    """
    visited = set()
    stack = list(particle.getParents())
    while stack:
        parent = stack.pop()
        uid = parent.getObjectID().index
        if uid in visited:
            continue
        visited.add(uid)
        if abs(parent.getPDG()) == PI0_PDG:
            return True
        stack.extend(parent.getParents())
    return False

# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# Each loader returns a dict of 1D numpy arrays, one entry per photon.
# All selections and region cuts are applied afterward via boolean masks.
# ─────────────────────────────────────────────────────────────────────────────
def load_file(path, label, max_events=None, has_fastsim_flag=False):
    """
    Generic loader for generator and sim files.

    Returns dict with keys:
        theta      [deg]
        phi        [deg]
        energy     [GeV]
        genstat    int
        is_pi0     bool
        is_fastsim bool  (always False unless has_fastsim_flag=True)
    """
    reader = root_io.Reader(path)
    events = reader.get("events")

    theta_list    = []
    phi_list      = []
    energy_list   = []
    genstat_list  = []
    is_pi0_list   = []
    is_fastsim_list = []

    n_events = 0
    for event in events:
        if max_events is not None and n_events >= max_events:
            break
        n_events += 1

        for p in event.get("MCParticles"):
            if p.getPDG() != PHOTON_PDG:
                continue

            p4    = edm4hep.utils.p4(p)
            theta = np.degrees(float(p4.theta()))
            phi   = np.degrees(float(p4.phi()))

            theta_list.append(theta)
            phi_list.append(phi)
            energy_list.append(p.getEnergy())
            genstat_list.append(p.getGeneratorStatus())
            is_pi0_list.append(has_pi0_ancestor(p))
            is_fastsim_list.append(
                bool(p.isHandledByFastSim()) if has_fastsim_flag else False
            )

    data = {
        "theta"     : np.array(theta_list),
        "phi"       : np.array(phi_list),
        "energy"    : np.array(energy_list),
        "genstat"   : np.array(genstat_list, dtype=int),
        "is_pi0"    : np.array(is_pi0_list,  dtype=bool),
        "is_fastsim": np.array(is_fastsim_list, dtype=bool),
    }

    n_photons = len(data["theta"])
    n_fastsim = np.sum(data["is_fastsim"]) if has_fastsim_flag else 0
    print(f"  {label}: {n_events} events, {n_photons} MC photons"
          + (f", {n_fastsim} isHandledByFastSim" if has_fastsim_flag else ""))
    return data

# ─────────────────────────────────────────────────────────────────────────────
# REGION MASKS
# ─────────────────────────────────────────────────────────────────────────────
def region_mask(theta_arr, region):
    if region == "full":
        return np.ones(len(theta_arr), dtype=bool)
    elif region == "barrel":
        return (theta_arr >= BARREL_MIN) & (theta_arr <= BARREL_MAX)
    elif region == "endcap":
        return (theta_arr <= ENDCAP_MAX_LOW) | (theta_arr >= ENDCAP_MIN_HIGH)
    else:
        raise ValueError(f"Unknown region: {region}")

# ─────────────────────────────────────────────────────────────────────────────
# SELECTION DEFINITIONS
# Each entry: (key, label_for_title, mask_fn, split_cc)
# split_cc=True  → CC3 plotted as two histograms (fast sim + G4-handled)
# split_cc=False → CC3 plotted as single total histogram
# ─────────────────────────────────────────────────────────────────────────────
def make_mask(data, genstat=None, emin=None, pi0_only=False):
    mask = np.ones(len(data["theta"]), dtype=bool)
    if pi0_only:
        mask &= data["is_pi0"]
    if genstat is not None:
        mask &= data["genstat"] == genstat
    if emin is not None:
        mask &= data["energy"] >= emin
    return mask

SELECTIONS = [
    # (key,                label,                                        genstat, emin,  pi0,   split_cc)
    ("all",               "All photons",                                  None,   None,  False, False),
    ("gs1",               "genStat=1",                                    1,      None,  False, True),
    ("gs1_e10",           "genStat=1, E≥10 GeV",                         1,      E_THRESHOLD, False, True),
    ("pi0",               "π⁰ daughters",                                 None,   None,  True,  False),
    ("pi0_gs1",           "π⁰ daughters, genStat=1",                     1,      None,  True,  False),
    ("pi0_gs1_e10",       "π⁰ daughters, genStat=1, E≥10 GeV",           1,      E_THRESHOLD, True, True),
]

REGIONS = [
    ("full",   "Full detector"),
    ("barrel", f"Barrel {BARREL_MIN:.0f}°–{BARREL_MAX:.0f}°"),
    ("endcap", f"Endcap 0°–{ENDCAP_MAX_LOW:.0f}° + {ENDCAP_MIN_HIGH:.0f}°–180°"),
]

# ─────────────────────────────────────────────────────────────────────────────
# BINNING  (physically motivated, consistent across all plots)
# θ: 36 bins of 5° each — ILD ECal resolution ~a few deg in theta
# φ: 36 bins of 10° each — should be flat for signal, easy to spot structure
# ─────────────────────────────────────────────────────────────────────────────
BINS_THETA = np.linspace(0,   180, 37)   # 36 bins × 5°
BINS_PHI   = np.linspace(-180, 180, 37)  # 36 bins × 10°

# ─────────────────────────────────────────────────────────────────────────────
# PLOT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def plot_angle(angle, gen_vals, g4_vals, cc_vals, cc_fs_vals, cc_g4_vals,
               bins, xlabel, title, filename, split_cc):
    """
    angle     : 'theta' or 'phi'
    *_vals    : 1D arrays of angle values after all masks applied
    split_cc  : if True, plot CC fast sim and CC G4 separately instead of CC total
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(gen_vals, bins=bins, histtype="step", color=C_GEN,
            linewidth=2, label=f"Gen (N={len(gen_vals)})")
    ax.hist(g4_vals,  bins=bins, histtype="step", color=C_G4,
            linewidth=2, label=f"G4 (N={len(g4_vals)})")

    if split_cc:
        ax.hist(cc_fs_vals, bins=bins, histtype="step", color=C_CC_FS,
                linewidth=2, linestyle="--",
                label=f"CC3 fast sim (N={len(cc_fs_vals)})")
        ax.hist(cc_g4_vals, bins=bins, histtype="step", color=C_CC_G4,
                linewidth=2, linestyle="-.",
                label=f"CC3 G4-handled (N={len(cc_g4_vals)})")
    else:
        ax.hist(cc_vals, bins=bins, histtype="step", color=C_CC,
                linewidth=2, linestyle="--",
                label=f"CC3 (N={len(cc_vals)})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Photons / bin")
    ax.set_title(title)
    ax.legend(framealpha=0.5)

    # Barrel / endcap region markers on theta plots
    if angle == "theta":
        for xval, lbl in [(BARREL_MIN, "barrel"), (BARREL_MAX, ""),
                          (ENDCAP_MAX_LOW, "endcap"), (ENDCAP_MIN_HIGH, "")]:
            ax.axvline(xval, color="gray", linewidth=0.8, linestyle=":")
        # Label once
        ax.axvline(BARREL_MIN,     color="gray", linewidth=0.8, linestyle=":",
                   label=f"Barrel {BARREL_MIN:.0f}°–{BARREL_MAX:.0f}°")

    ax.text(0.01, 1.01, "ILD sim V2 — 1k events (preliminary)",
            transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/{filename}", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {filename}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  Loading files")
print("="*65)
gen_data = load_file(GEN_FILE, "Generator    ", max_events=MAX_GEN_EVENTS, has_fastsim_flag=False)
g4_data  = load_file(G4_FILE,  "Geant4       ", max_events=None,           has_fastsim_flag=False)
cc_data  = load_file(CC_FILE,  "CaloClouds3  ", max_events=None,           has_fastsim_flag=True)

print("\n" + "="*65)
print("  Generating plots")
print("="*65)

for sel_key, sel_label, genstat, emin, pi0_only, split_cc in SELECTIONS:
    # Build masks
    gen_mask = make_mask(gen_data, genstat=genstat, emin=emin, pi0_only=pi0_only)
    g4_mask  = make_mask(g4_data,  genstat=genstat, emin=emin, pi0_only=pi0_only)
    cc_mask  = make_mask(cc_data,  genstat=genstat, emin=emin, pi0_only=pi0_only)

    # CC split masks
    cc_fs_mask = cc_mask &  cc_data["is_fastsim"]
    cc_g4_mask = cc_mask & ~cc_data["is_fastsim"]

    for reg_key, reg_label in REGIONS:
        # Region masks
        gen_reg = region_mask(gen_data["theta"], reg_key)
        g4_reg  = region_mask(g4_data["theta"],  reg_key)
        cc_reg  = region_mask(cc_data["theta"],  reg_key)

        final_gen    = gen_mask & gen_reg
        final_g4     = g4_mask  & g4_reg
        final_cc     = cc_mask  & cc_reg
        final_cc_fs  = cc_fs_mask & cc_reg
        final_cc_g4  = cc_g4_mask & cc_reg

        for angle, bins, xlabel_str in [
            ("theta", BINS_THETA, "MC Photon θ [deg]"),
            ("phi",   BINS_PHI,   "MC Photon φ [deg]"),
        ]:
            title = f"{angle.upper()} | {sel_label} | {reg_label} | sim V2"
            fname = f"{angle}_{sel_key}_{reg_key}.png"

            plot_angle(
                angle      = angle,
                gen_vals   = gen_data[angle][final_gen],
                g4_vals    = g4_data[angle][final_g4],
                cc_vals    = cc_data[angle][final_cc],
                cc_fs_vals = cc_data[angle][final_cc_fs],
                cc_g4_vals = cc_data[angle][final_cc_g4],
                bins       = bins,
                xlabel     = xlabel_str,
                title      = title,
                filename   = fname,
                split_cc   = split_cc,
            )

print(f"\n  All plots saved to: {PLOT_DIR}")
print(f"  Total plots: {len(SELECTIONS) * len(REGIONS) * 2}")
