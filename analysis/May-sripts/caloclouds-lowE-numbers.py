#!/usr/bin/env python3
"""
analysis_v3.py — CC3 vs Geant4 photon shower analysis
Samples : photon gun, 1-10 GeV uniform momentum, ILD_l5_v02, barrel direction (1,0,0)
Mask    : isHandledByFastSim==True AND non-empty ECal (CC3 files)
          non-empty ECal only (G4 file)
Author  : alimuham @ DESY
"""

import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
DATA_DIR = "/data/dust/user/alimuham/thesis/cc3_low_E_data"
OUT_DIR  = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/cc3_low_E_plots"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "G4"          : os.path.join(DATA_DIR, "photon_gun_1to10GeV_g4.edm4hep.root"),
    "CC3 original": os.path.join(DATA_DIR, "photon_gun_1to10GeV_cc3_original.edm4hep.root"),
    "CC3 630K"    : os.path.join(DATA_DIR, "photon_gun_1to10GeV_cc3_630K_test.edm4hep.root"),
}

COLORS = {
    "G4"          : "black",
    "CC3 original": "royalblue",
    "CC3 630K"    : "firebrick",
}

ECAL_COLLS = [
    "ECalBarrelSiHitsEven",
    "ECalBarrelSiHitsOdd",
    "ECalBarrelScHitsEven",
    "ECalBarrelScHitsOdd",
]

# 1 GeV-wide bins over the full gun range
E_EDGES   = np.arange(1.0, 10.5, 1.0)   # [1,2,...,10]
E_CENTERS = 0.5 * (E_EDGES[:-1] + E_EDGES[1:])

IS_CC3 = lambda label: "CC3" in label


# ─────────────────────────────────────────────
#  MC particle selection
#  Physics reason: with enableDetailedShowerMode=True, secondary
#  particles (e+/e- from pair production, brem photons) are stored
#  in MCParticles. We must select the gun photon explicitly:
#    PDG == 22  AND  generatorStatus == 1
#  to avoid accidentally using a secondary as the energy reference.
# ─────────────────────────────────────────────
def find_gun_photon(mc_particles):
    for p in mc_particles:
        if abs(p.getPDG()) == 22 and p.getGeneratorStatus() == 1:
            return p
    return None   # should never happen for a clean gun file

def gun_momentum(p):
    mom = p.getMomentum()
    return math.sqrt(mom.x**2 + mom.y**2 + mom.z**2)   # |p| in GeV


# ─────────────────────────────────────────────
#  File reader
# ─────────────────────────────────────────────
def process_file(label, path):
    import podio.root_io as rio

    if not os.path.exists(path):
        print(f"[SKIP] {label}: not found at {path}")
        return None

    print(f"\n[READ] {label}")
    r = rio.Reader(path)

    out = dict(
        gun_p      = [],
        total_edep = [],
        n_hits     = [],
        is_empty   = [],
        is_fastsim = [],
        n_events   = 0,
        # 2x2 accounting counters (filled after loop)
    )

    for frame in r.get("events"):
        out["n_events"] += 1

        # ── Gun photon ───────────────────────────────────────────────
        mc      = list(frame.get("MCParticles"))
        photon  = find_gun_photon(mc)
        if photon is None:
            # Pathological event — skip silently, record as empty
            out["gun_p"].append(np.nan)
            out["total_edep"].append(0.0)
            out["n_hits"].append(0)
            out["is_empty"].append(True)
            out["is_fastsim"].append(False)
            continue

        out["gun_p"].append(gun_momentum(photon))

        # ── Fast-sim flag ────────────────────────────────────────────
        if IS_CC3(label):
            try:
                fs = photon.isHandledByFastSim()
            except Exception:
                fs = False
        else:
            fs = False          # G4 has no fast-sim flag by definition
        out["is_fastsim"].append(fs)

        # ── ECal hits ────────────────────────────────────────────────
        hits = []
        for cname in ECAL_COLLS:
            try:
                hits.extend(list(frame.get(cname)))
            except Exception:
                pass

        n    = len(hits)
        edep = sum(h.getEnergy() for h in hits)
        out["n_hits"].append(n)
        out["total_edep"].append(edep)
        out["is_empty"].append(n == 0)

    # Convert to numpy
    for k in ("gun_p", "total_edep", "n_hits"):
        out[k] = np.array(out[k], dtype=float)
    out["is_empty"]   = np.array(out["is_empty"],   dtype=bool)
    out["is_fastsim"] = np.array(out["is_fastsim"], dtype=bool)

    _summarise(label, out)
    return out


def _summarise(label, d):
    n   = d["n_events"]
    fs  = d["is_fastsim"]
    emp = d["is_empty"]
    print(f"  Total events          : {n}")
    print(f"  Empty ECal            : {emp.sum():5d}  ({emp.mean()*100:.1f}%)")
    if IS_CC3(label):
        print(f"  isHandledByFastSim    : {fs.sum():5d}  ({fs.mean()*100:.1f}%)")
        # 2x2 table
        ff =  fs &  ~emp
        fe =  fs &   emp
        nf = ~fs &  ~emp
        ne = ~fs &   emp
        print(f"\n  2×2 breakdown (fastsim × ECal empty):")
        print(f"    FastSim=T, ECal non-empty : {ff.sum():5d}  ← analysis sample")
        print(f"    FastSim=T, ECal empty     : {fe.sum():5d}  ← CC3 fired but produced nothing")
        print(f"    FastSim=F, ECal non-empty : {nf.sum():5d}  ← G4 fallback, shower present")
        print(f"    FastSim=F, ECal empty     : {ne.sum():5d}  ← G4 fallback, no shower")
    else:
        print(f"  Non-empty (analysis)  : {(~emp).sum():5d}")
    mask = get_mask(label, d)
    sf = (d["total_edep"][mask] / d["gun_p"][mask]).mean() if mask.sum() > 0 else np.nan
    print(f"  Mean sampling fraction: {sf:.4f}  (expect ~0.02–0.05)")
    print(f"  Analysis events (mask): {mask.sum()}")


def get_mask(label, d):
    """Return the event mask appropriate for shower-quality analysis."""
    if IS_CC3(label):
        # Only events CC3 actually handled AND that have a non-empty ECal
        return d["is_fastsim"] & ~d["is_empty"]
    else:
        # G4: all non-empty ECal events
        return ~d["is_empty"]


# ─────────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────────
def _savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")


# ─────────────────────────────────────────────
#  Plot 1 — 2×2 accounting bar chart
#  Physics question: where do the empty ECal events come from?
#  Is CC3 failing to generate showers, or is G4 fallback producing empties?
# ─────────────────────────────────────────────
def plot_accounting(data):
    cc3_labels = [l for l in data if IS_CC3(l) and data[l] is not None]
    fig, axes = plt.subplots(1, len(cc3_labels), figsize=(5*len(cc3_labels), 5), sharey=True)
    if len(cc3_labels) == 1:
        axes = [axes]

    cats   = ["FS=T\nECal≠∅", "FS=T\nECal=∅", "FS=F\nECal≠∅", "FS=F\nECal=∅"]
    colors = ["#2a9d8f", "#e76f51", "#457b9d", "#adb5bd"]

    for ax, label in zip(axes, cc3_labels):
        d   = data[label]
        fs  = d["is_fastsim"]
        emp = d["is_empty"]
        counts = np.array([
            ( fs & ~emp).sum(),
            ( fs &  emp).sum(),
            (~fs & ~emp).sum(),
            (~fs &  emp).sum(),
        ])
        fracs = counts / d["n_events"]
        bars  = ax.bar(cats, fracs, color=colors, edgecolor="black", width=0.55)
        for bar, f, c in zip(bars, fracs, counts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    f + 0.012, f"{f:.3f}\n(N={c})",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Fraction of all events")
        ax.set_xlabel("FastSim flag × ECal status")

    fig.suptitle(
        "CC3 event accounting — where do the empty ECal events come from?\n"
        "FS=T: isHandledByFastSim==True | ~5k events per file | barrel gun (1,0,0)",
        fontsize=10
    )
    fig.tight_layout()
    _savefig(fig, "01_accounting_2x2.png")


# ─────────────────────────────────────────────
#  Plot 2 — Total ECal deposited energy spectrum
#  Physics question: does CC3 reproduce the overall E_dep distribution?
#  Expected: peak at ~2–5% of gun energy. Log x-axis because deposits
#  span ~20 MeV (1 GeV gun) to ~500 MeV (10 GeV gun) — >1 order of magnitude.
#  Normalised: event counts differ between files after masking.
#  Stating this explicitly in the title.
# ─────────────────────────────────────────────
def plot_edep_spectrum(data):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Log-spaced bins from 1 MeV to 1 GeV (in GeV units)
    bins = np.logspace(np.log10(0.001), np.log10(1.0), 80)

    for label, d in data.items():
        if d is None: continue
        mask = get_mask(label, d)
        edep = d["total_edep"][mask]
        ax.hist(edep, bins=bins, density=True, histtype="step",
                linewidth=1.8, color=COLORS[label],
                label=f"{label}  (N={mask.sum()})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total ECal barrel deposited energy [GeV]")
    ax.set_ylabel("Probability density [1/GeV]  (normalised)")
    ax.set_title(
        "Total ECal barrel energy deposit spectrum — all 4 collections summed\n"
        "CC3: isHandledByFastSim AND non-empty | G4: non-empty | Normalised (N differs)\n"
        "Log x-axis: deposits span ~20 MeV–500 MeV across 1–10 GeV gun range",
        fontsize=9
    )
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "02_edep_spectrum.png")


# ─────────────────────────────────────────────
#  Plot 3 — Energy response & resolution vs gun energy
#  Physics question:
#    Panel A: Does CC3 reproduce mean(E_dep)/E_gun (sampling fraction)?
#             Systematic offset = energy miscalibration.
#    Panel B: Does CC3 reproduce σ(E_dep)/⟨E_dep⟩ (relative resolution)?
#             This is the standard calorimeter figure of merit.
#             Expected shape: falls as ~a/√E (stochastic term dominates).
#  Binning: 1 GeV wide — physically motivated by flat gun spectrum,
#           ~equal statistics per bin, consistent across all three files.
#  NOT normalised: we are plotting per-bin means, error bars show
#  standard error on the mean — absolute comparison is the point.
# ─────────────────────────────────────────────
def plot_energy_response(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, d in data.items():
        if d is None: continue
        mask = get_mask(label, d)
        gun  = d["gun_p"][mask]
        edep = d["total_edep"][mask]

        means, errs, resols, resol_errs = [], [], [], []

        for lo, hi in zip(E_EDGES[:-1], E_EDGES[1:]):
            sel = (gun >= lo) & (gun < hi)
            if sel.sum() < 15:
                means.append(np.nan); errs.append(np.nan)
                resols.append(np.nan); resol_errs.append(np.nan)
                continue

            e_bin  = edep[sel]
            mu     = e_bin.mean()
            sigma  = e_bin.std(ddof=1)
            n      = sel.sum()

            # Sampling fraction = mean(E_dep) / E_gun_centre
            # Use bin centre as E_gun proxy (flat distribution → centre is unbiased)
            e_gun_centre = 0.5 * (lo + hi)
            sf = mu / e_gun_centre

            # Standard error on the mean sampling fraction
            sf_err = (sigma / e_gun_centre) / np.sqrt(n)

            # Resolution = σ(E_dep) / ⟨E_dep⟩   (standard calorimeter definition)
            resol     = sigma / mu if mu > 0 else np.nan
            # Error propagation (Gaussian approx):
            # δ(σ/μ) ≈ (σ/μ) * sqrt( 1/(2*(n-1)) + (δμ/μ)^2 )
            resol_err = resol * np.sqrt(1.0/(2*(n-1)) + (sigma/mu/np.sqrt(n))**2) if mu > 0 else np.nan

            means.append(sf);       errs.append(sf_err)
            resols.append(resol);   resol_errs.append(resol_err)

        kw = dict(color=COLORS[label], linewidth=1.5, markersize=5, capsize=3)
        axes[0].errorbar(E_CENTERS, means,  yerr=errs,       fmt="o-", label=label, **kw)
        axes[1].errorbar(E_CENTERS, resols, yerr=resol_errs, fmt="s--", label=label, **kw)

    axes[0].set_xlabel("Gun momentum |p| = √(px²+py²+pz²) [GeV]")
    axes[0].set_ylabel("Mean  E_dep / E_gun  (sampling fraction)")
    axes[0].set_title(
        "Energy response (sampling fraction) vs gun energy\n"
        "CC3: isHandledByFastSim AND non-empty | G4: non-empty\n"
        "Error bars: standard error on the mean | 1 GeV bins"
    )
    axes[0].legend()
    axes[0].set_xlim(1, 10)

    axes[1].set_xlabel("Gun momentum |p| [GeV]")
    axes[1].set_ylabel("σ(E_dep) / ⟨E_dep⟩  (relative resolution)")
    axes[1].set_title(
        "Energy resolution σ/⟨E⟩ vs gun energy\n"
        "Standard calorimeter definition | expect ~a/√E falling curve\n"
        "Error bars: propagated from σ and N"
    )
    axes[1].legend()
    axes[1].set_xlim(1, 10)

    fig.suptitle(
        "ILD ECal barrel — photon gun 1–10 GeV | barrel direction (1,0,0) | ILD_l5_v02",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
    _savefig(fig, "03_energy_response_resolution.png")


# ─────────────────────────────────────────────
#  Plot 4 — Hit multiplicity distribution
#  Physics question: does CC3 produce the right number of active cells?
#  This probes spatial shower spread independent of total energy.
#  Normalised: event counts differ after masking — raw counts would
#  make the comparison misleading if one model triggers less often.
# ─────────────────────────────────────────────
def plot_hit_multiplicity(data):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Determine bin range from the 1st–99th percentile across all datasets
    all_hits = np.concatenate([
        d["n_hits"][get_mask(l, d)]
        for l, d in data.items() if d is not None
    ])
    lo_pct = int(np.percentile(all_hits, 1))
    hi_pct = int(np.percentile(all_hits, 99)) + 1
    bins = np.linspace(lo_pct, hi_pct, 80)

    for label, d in data.items():
        if d is None: continue
        mask = get_mask(label, d)
        ax.hist(d["n_hits"][mask], bins=bins, density=True,
                histtype="step", linewidth=1.8,
                color=COLORS[label], label=f"{label}  (N={mask.sum()})")

    ax.set_xlabel("Total ECal barrel hit count per event (all 4 collections)")
    ax.set_ylabel("Probability density  (normalised)")
    ax.set_title(
        "Hit multiplicity distribution — proxy for shower lateral spread\n"
        "CC3: isHandledByFastSim AND non-empty | G4: non-empty | Normalised\n"
        "Bins: 1st–99th percentile of combined sample, 80 equal-width bins"
    )
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "04_hit_multiplicity.png")


# ─────────────────────────────────────────────
#  Plot 5 — Hit multiplicity vs gun energy (mean + std)
#  Physics question: does CC3's hit count scale correctly with energy?
#  In a real shower, N_hits grows roughly linearly with E_gun.
# ─────────────────────────────────────────────
def plot_hits_vs_energy(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, d in data.items():
        if d is None: 
            continue

        mask  = get_mask(label, d)
        gun   = d["gun_p"][mask]
        nhits = d["n_hits"][mask]

        means, stds, se = [], [], []

        for lo, hi in zip(E_EDGES[:-1], E_EDGES[1:]):
            sel = (gun >= lo) & (gun < hi)
            n   = sel.sum()

            if n < 15:
                means.append(np.nan)
                stds.append(np.nan)
                se.append(np.nan)
                continue

            mu = nhits[sel].mean()
            s  = nhits[sel].std(ddof=1)

            means.append(mu)
            stds.append(s)
            se.append(s / np.sqrt(n))

        kw = dict(color=COLORS[label], linewidth=1.5, markersize=5, capsize=3)
        axes[0].errorbar(E_CENTERS, means, yerr=se,   fmt="o-",  label=label, **kw)
        axes[1].errorbar(E_CENTERS, stds,  yerr=None, fmt="s--", label=label, **kw)

    for ax, ylabel, title_extra in zip(
        axes,
        ["Mean N_hits per event", "Std N_hits per event"],
        ["Mean hit multiplicity vs gun energy\nExpect roughly linear growth",
         "Std of hit multiplicity vs gun energy\nCaptured shower fluctuations"]
    ):
        ax.set_xlabel("Gun momentum |p| [GeV]")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{title_extra}\n"
            "CC3: isHandledByFastSim AND non-empty | G4: non-empty | 1 GeV bins"
        )
        ax.legend()
        ax.set_xlim(1, 10)

    fig.tight_layout()
    _savefig(fig, "05_hits_vs_energy.png")

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  CC3 vs G4 photon shower analysis — v3")
    print("  Steering: G4 and CC3 files, 1-10 GeV uniform gun, barrel")
    print("  Mask: isHandledByFastSim AND non-empty ECal (CC3)")
    print("        non-empty ECal only (G4)")
    print("=" * 65)

    data = {label: process_file(label, path) for label, path in FILES.items()}

    print("\n\nGenerating plots → ", OUT_DIR)
    plot_accounting(data)
    plot_edep_spectrum(data)
    plot_energy_response(data)
    plot_hit_multiplicity(data)
    plot_hits_vs_energy(data)

    print("\n[DONE]")
