"""
energy_distributions.py
========================
MC truth photon energy distributions.
Compares GEN input, Geant4 sim, and CaloClouds3 sim.

Plots produced (all log-y, raw counts, 10 GeV threshold line):
  1. All photons                              — GEN, G4, CC3 total
  2. genStatus == 1                           — GEN, G4, CC3 total
  3. genStatus==1, pi0 daughter               — GEN, G4, CC3 split (FS + G4-handled)
                                                + bottom panel: fast-sim fraction
  4. genStatus==1, not ISR                    — GEN, G4, CC3 split (FS + G4-handled)
                                                + bottom panel: fast-sim fraction
  5. genStatus==1, not ISR, NOT fast-simmed   — CC3 only + G4 reference overlay

All histograms: raw counts, NOT normalized.
GEN file capped at 100k events.

Speed: single pass per file, all masking done in numpy after loading.

Run:
    source ~/source.sh
    python3 energy_distributions.py
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

PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/energy_distributions_100k")
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_EVENTS = 100_000
E_THRESH   = 10.0
PHOTON_PDG = 22
PI0_PDG    = 111

# ─── Colors ───────────────────────────────────────────────────────────────────
C_GEN    = "#333333"   # dark gray  — generator
C_G4     = "#2166ac"   # blue       — Geant4
C_CC3    = "#d6604d"   # red        — CC3 total
C_CC_FS  = "#f4a582"   # orange     — CC3 fast-sim handled
C_CC_G4  = "#4dac26"   # green      — CC3 G4-handled
C_MISSED = "#7b2d8b"   # purple     — missed by fast sim
C_FRAC   = "#d6604d"   # red        — fraction line (same as CC3 total)

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
# 2 GeV bins from 0 to 130 GeV.
# ECal energy resolution ~15%/sqrt(E), so at 10 GeV ~0.5 GeV.
# 2 GeV bins are safely above resolution, give good statistics per bin.
BINS_E = np.arange(0, 132, 2)   # 66 bins × 2 GeV


# ─── Loader ───────────────────────────────────────────────────────────────────
def load_file(label, path, is_cc3=False, max_events=MAX_EVENTS):
    """
    Single pass through the file. Stores all photon-level quantities as lists,
    converts to numpy arrays at the end. No repeated reads.
    """
    print(f"  Loading {label} ...")
    reader = root_io.Reader(path)

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

        # Build pi0-daughter index set once per event — O(n_particles), O(1) lookup
        pi0_children = set()
        for p in particles:
            if abs(p.getPDG()) == PI0_PDG:
                for child in p.getDaughters():
                    pi0_children.add(child.getObjectID().index)

        for p in particles:
            if p.getPDG() != PHOTON_PDG:
                continue

            parent_pdgs = {abs(par.getPDG()) for par in p.getParents()}
            idx         = p.getObjectID().index

            energy_arr.append(p.getEnergy())
            genstat_arr.append(p.getGeneratorStatus())
            is_pi0_arr.append(idx in pi0_children)
            is_isr_arr.append(11 in parent_pdgs)
            is_fs_arr.append(bool(p.isHandledByFastSim()) if is_cc3 else False)

    d = {
        "energy" : np.array(energy_arr),
        "genstat": np.array(genstat_arr, dtype=int),
        "is_pi0" : np.array(is_pi0_arr,  dtype=bool),
        "is_isr" : np.array(is_isr_arr,  dtype=bool),
        "is_fs"  : np.array(is_fs_arr,   dtype=bool),
    }

    n_ph = len(d["energy"])
    n_fs = int(np.sum(d["is_fs"])) if is_cc3 else 0
    print(f"    {n_events} events | {n_ph} photons"
          + (f" | {n_fs} fast-sim handled" if is_cc3 else ""))
    return d, n_events


# ─── Selection helper ─────────────────────────────────────────────────────────
def sel(d, genstat=None, pi0=False, not_isr=False, missed=False):
    m = np.ones(len(d["energy"]), dtype=bool)
    if genstat is not None : m &= d["genstat"] == genstat
    if pi0                 : m &= d["is_pi0"]
    if not_isr             : m &= ~d["is_isr"]
    if missed              : m &= ~d["is_fs"]
    return m


# ─── Fast-sim fraction panel helper ──────────────────────────────────────────
def add_fraction_panel(ax, cc3_total_e, cc3_fs_e, bins):
    """
    Draw fast-sim fraction (cc3_fs / cc3_total) per energy bin onto ax.

    Physics question answered: at each energy, what fraction of CC3-eligible
    photons were actually handled by the fast-sim model vs. falling back to G4?

    Denominator = CC3 total for this selection (NOT all CC3 photons).
    Numerator   = CC3 fast-sim handled within this selection.

    Error bars: binomial (Wilson-style approximation via sqrt(p(1-p)/n)).
    Bins with zero total entries are skipped (fraction undefined).
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    counts_total, _ = np.histogram(cc3_total_e, bins=bins)
    counts_fs,    _ = np.histogram(cc3_fs_e,    bins=bins)

    # Avoid division by zero — mask bins with no entries
    valid = counts_total > 0
    frac     = np.where(valid, counts_fs / counts_total, 0.0)
    frac_err = np.where(valid,
                        np.sqrt(frac * (1 - frac) / np.where(valid, counts_total, 1)),
                        0.0)

    ax.errorbar(bin_centers[valid], frac[valid], yerr=frac_err[valid],
                fmt="o", color=C_FRAC, markersize=4,
                linewidth=1.2, capsize=2,
                label="Fast-sim fraction (CC3 FS / CC3 total)")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":",  alpha=0.5,
               label="50% reference")
    ax.axvline(E_THRESH, color="black", linewidth=1.0, linestyle=":",
               alpha=0.7, label=f"{E_THRESH:.0f} GeV threshold")
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Fast-sim fraction")
    ax.legend(framealpha=0.5, fontsize=8, loc="lower right")


# ─── Plot helper ──────────────────────────────────────────────────────────────
def make_plot(title, filename,
              gen_e=None, g4_e=None,
              cc3_total_e=None,
              cc3_fs_e=None, cc3_g4_e=None,
              cc3_missed_e=None,
              g4_ref_e=None,
              n_gen=None, n_g4=None, n_cc3=None,
              show_fraction=False):
    """
    Flexible energy histogram.

    show_fraction=True  →  two-panel figure (2:1 height ratio).
                           Bottom panel shows bin-by-bin fast-sim fraction.
                           Requires cc3_fs_e and cc3_total_e to be provided.

    show_fraction=False →  single-panel figure (original behaviour).

    Panel contents:
    - gen_e, g4_e            : step histograms (GEN and G4)
    - cc3_total_e            : dashed step histogram (CC3 all)
    - cc3_fs_e + cc3_g4_e   : split CC3 mode (filled fast-sim + dashed G4-handled)
    - cc3_missed_e           : purple step (plot 5)
    - g4_ref_e               : gray filled reference (plot 5 overlay, drawn first)
    """
    if show_fraction:
        # Two-panel: top (main histograms, 2/3) + bottom (fraction, 1/3)
        fig, (ax, ax_frac) = plt.subplots(
            2, 1, figsize=(10, 8),
            gridspec_kw={"height_ratios": [2, 1]},
            sharex=True
        )
        fig.subplots_adjust(hspace=0.08)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax_frac = None

    # ── G4 reference overlay (plot 5) — drawn first so it sits behind ────────
    if g4_ref_e is not None:
        ax.hist(g4_ref_e, bins=BINS_E, histtype="stepfilled",
                color=C_G4, alpha=0.15,
                label=f"G4 reference: gen1, not ISR (N={len(g4_ref_e)})")
        ax.hist(g4_ref_e, bins=BINS_E, histtype="step",
                color=C_G4, linewidth=1.5, linestyle="--", alpha=0.6)

    if gen_e is not None:
        ax.hist(gen_e, bins=BINS_E, histtype="step",
                color=C_GEN, linewidth=2,
                label=f"GEN (N={len(gen_e)}, {n_gen} ev)")

    if g4_e is not None:
        ax.hist(g4_e, bins=BINS_E, histtype="step",
                color=C_G4, linewidth=2,
                label=f"G4 (N={len(g4_e)}, {n_g4} ev)")

    if cc3_total_e is not None:
        ax.hist(cc3_total_e, bins=BINS_E, histtype="step",
                color=C_CC3, linewidth=2, linestyle="--",
                label=f"CC3 total (N={len(cc3_total_e)}, {n_cc3} ev)")

    if cc3_fs_e is not None:
        ax.hist(cc3_fs_e, bins=BINS_E, histtype="stepfilled",
                color=C_CC_FS, alpha=0.5,
                label=f"CC3 fast-sim (N={len(cc3_fs_e)})")
        ax.hist(cc3_fs_e, bins=BINS_E, histtype="step",
                color=C_CC_FS, linewidth=1.5)

    if cc3_g4_e is not None:
        ax.hist(cc3_g4_e, bins=BINS_E, histtype="step",
                color=C_CC_G4, linewidth=2, linestyle="-.",
                label=f"CC3 G4-handled (N={len(cc3_g4_e)})")

    if cc3_missed_e is not None:
        ax.hist(cc3_missed_e, bins=BINS_E, histtype="step",
                color=C_MISSED, linewidth=2,
                label=f"CC3 missed by fast-sim (N={len(cc3_missed_e)}, {n_cc3} ev)")

    # 10 GeV threshold line — only on main panel (not fraction panel, which has its own)
    ax.axvline(E_THRESH, color="black", linewidth=1.2, linestyle=":",
               label=f"CC3 trigger threshold ({E_THRESH:.0f} GeV)")

    ax.set_ylabel("Photons / 2 GeV  [raw counts]")
    ax.set_yscale("log")
    ax.set_xlim(0, 130)
    ax.set_title(title)
    ax.legend(framealpha=0.5, fontsize=9)
    ax.text(0.01, 1.01,
            "ILD sim — 100k events  |  raw counts",
            transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

    # ── Fraction panel (only for plots 3 and 4) ───────────────────────────────
    if show_fraction and cc3_fs_e is not None and cc3_total_e is not None:
        add_fraction_panel(ax_frac, cc3_total_e, cc3_fs_e, BINS_E)
        ax_frac.set_xlabel("MC Photon Energy [GeV]")
    elif ax_frac is None:
        ax.set_xlabel("MC Photon Energy [GeV]")

    plt.tight_layout()
    out = f"{PLOT_DIR}/{filename}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {filename}")


# ─── Main ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Energy distributions — 100k events")
print("=" * 60 + "\n")

gen, n_gen = load_file("GEN (input)",    GEN_FILE, is_cc3=False)
g4,  n_g4  = load_file("G4  (full sim)", G4_FILE,  is_cc3=False)
cc3, n_cc3 = load_file("CC3 (fast sim)", CC3_FILE, is_cc3=True)

print()

# ─── Masks ────────────────────────────────────────────────────────────────────
# Plot 1: all photons (no mask)

# Plot 2: genStatus == 1
m_gen_2 = sel(gen, genstat=1)
m_g4_2  = sel(g4,  genstat=1)
m_cc3_2 = sel(cc3, genstat=1)

# Plot 3: genStatus==1, pi0 daughter
m_gen_3      = sel(gen, genstat=1, pi0=True)
m_g4_3       = sel(g4,  genstat=1, pi0=True)
m_cc3_3      = sel(cc3, genstat=1, pi0=True)
m_cc3_3_fs   = m_cc3_3 &  cc3["is_fs"]
m_cc3_3_g4h  = m_cc3_3 & ~cc3["is_fs"]

# Plot 4: genStatus==1, not ISR
m_gen_4      = sel(gen, genstat=1, not_isr=True)
m_g4_4       = sel(g4,  genstat=1, not_isr=True)
m_cc3_4      = sel(cc3, genstat=1, not_isr=True)
m_cc3_4_fs   = m_cc3_4 &  cc3["is_fs"]
m_cc3_4_g4h  = m_cc3_4 & ~cc3["is_fs"]

# Plot 5: genStatus==1, not ISR, NOT fast-simmed (CC3 only)
m_cc3_5      = sel(cc3, genstat=1, not_isr=True, missed=True)

# ─── Print summary counts ─────────────────────────────────────────────────────
print("  Selection summary:")
print(f"  {'Selection':<40} {'GEN':>8} {'G4':>8} {'CC3':>8}")
print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'All photons':<40} {len(gen['energy']):>8} {len(g4['energy']):>8} {len(cc3['energy']):>8}")
print(f"  {'genStatus==1':<40} {m_gen_2.sum():>8} {m_g4_2.sum():>8} {m_cc3_2.sum():>8}")
print(f"  {'genStatus==1, pi0 daughter':<40} {m_gen_3.sum():>8} {m_g4_3.sum():>8} {m_cc3_3.sum():>8}")
print(f"  {'  └─ CC3 fast-sim':<40} {'':>8} {'':>8} {m_cc3_3_fs.sum():>8}")
print(f"  {'  └─ CC3 G4-handled':<40} {'':>8} {'':>8} {m_cc3_3_g4h.sum():>8}")
print(f"  {'genStatus==1, not ISR':<40} {m_gen_4.sum():>8} {m_g4_4.sum():>8} {m_cc3_4.sum():>8}")
print(f"  {'  └─ CC3 fast-sim':<40} {'':>8} {'':>8} {m_cc3_4_fs.sum():>8}")
print(f"  {'  └─ CC3 G4-handled':<40} {'':>8} {'':>8} {m_cc3_4_g4h.sum():>8}")
print(f"  {'genStatus==1, not ISR, missed':<40} {'N/A':>8} {'N/A':>8} {m_cc3_5.sum():>8}")
print()

# ─── Plot 1: all photons ──────────────────────────────────────────────────────
make_plot(
    title    = "Energy — All MC photons | GEN vs G4 vs CC3",
    filename = "1_energy_all.png",
    gen_e    = gen["energy"],
    g4_e     = g4["energy"],
    cc3_total_e = cc3["energy"],
    n_gen=n_gen, n_g4=n_g4, n_cc3=n_cc3,
)

# ─── Plot 2: genStatus == 1 ───────────────────────────────────────────────────
make_plot(
    title    = "Energy — genStatus==1 | GEN vs G4 vs CC3",
    filename = "2_energy_gs1.png",
    gen_e    = gen["energy"][m_gen_2],
    g4_e     = g4["energy"][m_g4_2],
    cc3_total_e = cc3["energy"][m_cc3_2],
    n_gen=n_gen, n_g4=n_g4, n_cc3=n_cc3,
)

# ─── Plot 3: genStatus==1, pi0 daughter ──────────────────────────────────────
# Physics question for fraction panel:
# As a function of energy, what fraction of π⁰ daughter photons does CC3
# actually fast-simulate? Expected: ~1.0 above 10 GeV in barrel, dropping
# in dead zones.
make_plot(
    title    = ("Energy — genStatus==1, π⁰ daughter | GEN vs G4 vs CC3 split\n"
                "Bottom: fast-sim fraction = CC3 FS / CC3 total (this selection)"),
    filename = "3_energy_gs1_pi0.png",
    gen_e    = gen["energy"][m_gen_3],
    g4_e     = g4["energy"][m_g4_3],
    cc3_total_e = cc3["energy"][m_cc3_3],
    cc3_fs_e    = cc3["energy"][m_cc3_3_fs],
    cc3_g4_e    = cc3["energy"][m_cc3_3_g4h],
    n_gen=n_gen, n_g4=n_g4, n_cc3=n_cc3,
    show_fraction=True,
)

# ─── Plot 4: genStatus==1, not ISR ───────────────────────────────────────────
# Physics question for fraction panel:
# Same but for the broader ISR-rejected signal sample.
# Should look similar to plot 3 since pi0 daughters dominate this selection.
make_plot(
    title    = ("Energy — genStatus==1, not ISR | GEN vs G4 vs CC3 split\n"
                "Bottom: fast-sim fraction = CC3 FS / CC3 total (this selection)"),
    filename = "4_energy_gs1_notisr.png",
    gen_e    = gen["energy"][m_gen_4],
    g4_e     = g4["energy"][m_g4_4],
    cc3_total_e = cc3["energy"][m_cc3_4],
    cc3_fs_e    = cc3["energy"][m_cc3_4_fs],
    cc3_g4_e    = cc3["energy"][m_cc3_4_g4h],
    n_gen=n_gen, n_g4=n_g4, n_cc3=n_cc3,
    show_fraction=True,
)

# ─── Plot 5: missed by fast sim — CC3 only + G4 reference ────────────────────
make_plot(
    title    = ("Energy — genStatus==1, not ISR, NOT fast-simmed | CC3 only\n"
                "G4 reference: same selection without missed cut"),
    filename = "5_energy_missed.png",
    cc3_missed_e = cc3["energy"][m_cc3_5],
    g4_ref_e     = g4["energy"][m_g4_4],
    n_cc3=n_cc3, n_g4=n_g4,
)

print(f"\n  All 5 plots saved to: {PLOT_DIR}\n")
