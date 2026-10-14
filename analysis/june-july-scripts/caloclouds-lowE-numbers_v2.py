#!/usr/bin/env python3
"""
analysis_v4.py — CC3 vs Geant4 photon shower analysis
Samples : photon gun, 1-10 GeV uniform momentum, ILD_l5_v02, barrel direction (1,0,0)
Mask    : isHandledByFastSim==True AND non-empty ECal (CC3 files)
          non-empty ECal only (G4 file)
Changes from v3:
  - ECAL_COLLS now includes endcap collections (fixes ~20% false-empty rate)
  - Added Plot 6: z=0mm vs z=20mm E_dep overlay (G4 only)
  - Added Plot 7: longitudinal shower profile (mean E_dep vs ECAL layer)
  - Added Plot 8: transverse shower profile (mean E_dep vs r_perp from shower axis)
Author  : alimuham @ DESY
"""

import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
DATA_DIR   = "/data/dust/user/alimuham/thesis/cc3_low_E_data"
OUT_DIR    = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/cc3_low_E_plots"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "G4"          : os.path.join(DATA_DIR, "photon_gun_1to10GeV_g4_z20mm.edm4hep.root"),
    "CC3 original": os.path.join(DATA_DIR, "photon_gun_1to10GeV_cc3_original_z20mm.edm4hep.root"),
    "CC3 test"    : os.path.join(DATA_DIR, "photon_gun_1to10GeV_cc3_july7_undistilled_z20mm.edm4hep.root"),
}

# z=0mm files for the gun-origin comparison (G4 only)
FILES_Z0 = {
    "G4 z=0mm" : os.path.join(DATA_DIR, "photon_gun_1to10GeV_g4.edm4hep.root"),
}

COLORS = {
    "G4"          : "black",
    "CC3 original": "royalblue",
    "CC3 test"    : "firebrick",
    "G4 z=0mm"    : "dimgray",
}

# v3 had barrel only — now includes endcap (fixes false-empty events)
ECAL_COLLS = [
    "ECalBarrelSiHitsEven",
    "ECalBarrelSiHitsOdd",
    "ECalBarrelScHitsEven",
    "ECalBarrelScHitsOdd",
    "ECalEndcapSiHitsEven",
    "ECalEndcapSiHitsOdd",
    "ECalEndcapScHitsEven",
    "ECalEndcapScHitsOdd",
    "EcalEndcapRingCollection",
]

# ILD_l5_v02 ECAL barrel geometry constants (for longitudinal profile)
ECAL_INNER_R  = 1843.0   # mm — barrel inner face radius
ECAL_LAYER_DZ = 5.4      # mm — approximate Si+W layer pitch
N_LAYERS      = 30

E_EDGES   = np.arange(1.0, 10.5, 1.0)
E_CENTERS = 0.5 * (E_EDGES[:-1] + E_EDGES[1:])

IS_CC3 = lambda label: "CC3" in label


# ─────────────────────────────────────────────
#  MC particle selection
# ─────────────────────────────────────────────
def find_gun_photon(mc_particles):
    for p in mc_particles:
        if abs(p.getPDG()) == 22 and p.getGeneratorStatus() == 1:
            return p
    return None

def gun_momentum(p):
    mom = p.getMomentum()
    return math.sqrt(mom.x**2 + mom.y**2 + mom.z**2)


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
    )

    for frame in r.get("events"):
        out["n_events"] += 1

        mc     = list(frame.get("MCParticles"))
        photon = find_gun_photon(mc)
        if photon is None:
            out["gun_p"].append(np.nan)
            out["total_edep"].append(0.0)
            out["n_hits"].append(0)
            out["is_empty"].append(True)
            out["is_fastsim"].append(False)
            continue

        out["gun_p"].append(gun_momentum(photon))

        if IS_CC3(label):
            try:
                fs = photon.isHandledByFastSim()
            except Exception:
                fs = False
        else:
            fs = False
        out["is_fastsim"].append(fs)

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
        ff =  fs & ~emp
        fe =  fs &  emp
        nf = ~fs & ~emp
        ne = ~fs &  emp
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
    if IS_CC3(label):
        return d["is_fastsim"] & ~d["is_empty"]
    else:
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
        "FS=T: isHandledByFastSim==True | barrel+endcap ECal collections | barrel gun (1,0,0)",
        fontsize=10
    )
    fig.tight_layout()
    _savefig(fig, "01_accounting_2x2.png")


# ─────────────────────────────────────────────
#  Plot 2 — Total ECal deposited energy spectrum
# ─────────────────────────────────────────────
def plot_edep_spectrum(data):
    fig, ax = plt.subplots(figsize=(8, 5))
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
    ax.set_xlabel("Total ECal deposited energy [GeV]")
    ax.set_ylabel("Probability density [1/GeV]  (normalised)")
    ax.set_title(
        "Total ECal energy deposit spectrum — barrel + endcap collections\n"
        "CC3: isHandledByFastSim AND non-empty | G4: non-empty | Normalised\n"
        "Log x-axis: deposits span ~20 MeV–500 MeV across 1–10 GeV gun range",
        fontsize=9
    )
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "02_edep_spectrum.png")


# ─────────────────────────────────────────────
#  Plot 3 — Energy response & resolution vs gun energy
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

            e_bin        = edep[sel]
            mu           = e_bin.mean()
            sigma        = e_bin.std(ddof=1)
            n            = sel.sum()
            e_gun_centre = 0.5 * (lo + hi)
            sf           = mu / e_gun_centre
            sf_err       = (sigma / e_gun_centre) / np.sqrt(n)
            resol        = sigma / mu if mu > 0 else np.nan
            resol_err    = resol * np.sqrt(1.0/(2*(n-1)) + (sigma/mu/np.sqrt(n))**2) if mu > 0 else np.nan

            means.append(sf);     errs.append(sf_err)
            resols.append(resol); resol_errs.append(resol_err)

        kw = dict(color=COLORS[label], linewidth=1.5, markersize=5, capsize=3)
        axes[0].errorbar(E_CENTERS, means,  yerr=errs,       fmt="o-",  label=label, **kw)
        axes[1].errorbar(E_CENTERS, resols, yerr=resol_errs, fmt="s--", label=label, **kw)

    axes[0].set_xlabel("Gun momentum |p| [GeV]")
    axes[0].set_ylabel("Mean  E_dep / E_gun  (sampling fraction)")
    axes[0].set_title(
        "Energy response (sampling fraction) vs gun energy\n"
        "CC3: isHandledByFastSim AND non-empty | G4: non-empty\n"
        "Error bars: standard error on the mean | 1 GeV bins"
    )
    axes[0].legend(); axes[0].set_xlim(1, 10)

    axes[1].set_xlabel("Gun momentum |p| [GeV]")
    axes[1].set_ylabel("σ(E_dep) / ⟨E_dep⟩  (relative resolution)")
    axes[1].set_title(
        "Energy resolution σ/⟨E⟩ vs gun energy\n"
        "Standard calorimeter definition | expect ~a/√E falling curve\n"
        "Error bars: propagated from σ and N"
    )
    axes[1].legend(); axes[1].set_xlim(1, 10)

    fig.suptitle(
        "ILD ECal barrel+endcap — photon gun 1–10 GeV | dir (1,0,0) | ILD_l5_v02",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
    _savefig(fig, "03_energy_response_resolution.png")


# ─────────────────────────────────────────────
#  Plot 4 — Hit multiplicity distribution
# ─────────────────────────────────────────────
def plot_hit_multiplicity(data):
    fig, ax = plt.subplots(figsize=(8, 5))

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

    ax.set_xlabel("Total ECal hit count per event (barrel + endcap)")
    ax.set_ylabel("Probability density  (normalised)")
    ax.set_title(
        "Hit multiplicity distribution — proxy for shower lateral spread\n"
        "CC3: isHandledByFastSim AND non-empty | G4: non-empty | Normalised"
    )
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "04_hit_multiplicity.png")


# ─────────────────────────────────────────────
#  Plot 5 — Hit multiplicity vs gun energy
# ─────────────────────────────────────────────
def plot_hits_vs_energy(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, d in data.items():
        if d is None: continue
        mask  = get_mask(label, d)
        gun   = d["gun_p"][mask]
        nhits = d["n_hits"][mask]
        means, stds, se = [], [], []

        for lo, hi in zip(E_EDGES[:-1], E_EDGES[1:]):
            sel = (gun >= lo) & (gun < hi)
            n   = sel.sum()
            if n < 15:
                means.append(np.nan); stds.append(np.nan); se.append(np.nan)
                continue
            mu = nhits[sel].mean()
            s  = nhits[sel].std(ddof=1)
            means.append(mu); stds.append(s); se.append(s / np.sqrt(n))

        kw = dict(color=COLORS[label], linewidth=1.5, markersize=5, capsize=3)
        axes[0].errorbar(E_CENTERS, means, yerr=se,   fmt="o-",  label=label, **kw)
        axes[1].errorbar(E_CENTERS, stds,  yerr=None, fmt="s--", label=label, **kw)

    for ax, ylabel, title_extra in zip(
        axes,
        ["Mean N_hits per event", "Std N_hits per event"],
        ["Mean hit multiplicity vs gun energy\nExpect roughly linear growth",
         "Std of hit multiplicity vs gun energy"]
    ):
        ax.set_xlabel("Gun momentum |p| [GeV]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_extra}\nCC3: isHandledByFastSim AND non-empty | G4: non-empty | 1 GeV bins")
        ax.legend(); ax.set_xlim(1, 10)

    fig.tight_layout()
    _savefig(fig, "05_hits_vs_energy.png")


# ─────────────────────────────────────────────
#  Plot 6 — z=0mm vs z=20mm gun origin comparison (G4 only)
#  Physics question: does the gun start position shift the shower response?
#  It shouldn't — if it does, something geometric is wrong.
#  We overlay the total E_dep spectrum for both z positions, G4 only.
# ─────────────────────────────────────────────
def plot_z0_vs_z20(data, data_z0):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(0.001), np.log10(1.0), 80)

    # z=20mm (from main data dict, G4 only)
    d20 = data.get("G4")
    if d20 is not None:
        mask = get_mask("G4", d20)
        ax.hist(d20["total_edep"][mask], bins=bins, density=True,
                histtype="step", linewidth=2.0, color="black",
                label=f"G4  z=20mm  (N={mask.sum()})")

    # z=0mm
    for label, d in data_z0.items():
        if d is None: continue
        mask = get_mask(label, d)
        ax.hist(d["total_edep"][mask], bins=bins, density=True,
                histtype="step", linewidth=2.0, linestyle="--",
                color=COLORS.get(label, "dimgray"),
                label=f"{label}  (N={mask.sum()})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total ECal deposited energy [GeV]")
    ax.set_ylabel("Probability density [1/GeV]  (normalised)")
    ax.set_title(
        "Gun origin comparison: z=0mm vs z=20mm — G4 only\n"
        "Physics question: does gun start position shift the shower response?\n"
        "Solid: z=20mm | Dashed: z=0mm | Normalised | barrel+endcap ECal",
        fontsize=9
    )
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "06_z0_vs_z20_edep.png")


# ─────────────────────────────────────────────
#  Plot 7 — Longitudinal shower profile
#  Physics question: does CC3 place the shower maximum at the right depth?
#  Layer index computed from r = sqrt(x²+y²) for barrel hits.
#  ILD barrel inner face r=1843mm, pitch≈5.4mm, 30 layers.
#  For endcap hits r_barrel is not meaningful — we skip them here
#  (longitudinal profile is most cleanly defined for barrel-only photons).
# ─────────────────────────────────────────────
def plot_longitudinal_profile(data):
    import podio.root_io as rio
    fig, ax = plt.subplots(figsize=(9, 5))

    # Barrel-only collections for clean longitudinal profile
    barrel_colls = [
        "ECalBarrelSiHitsEven", "ECalBarrelSiHitsOdd",
        "ECalBarrelScHitsEven", "ECalBarrelScHitsOdd",
    ]

    for label, d in data.items():
        if d is None: continue
        mask = get_mask(label, d)
        layer_edep    = np.zeros(N_LAYERS)
        n_events_mask = 0
        events        = list(rio.Reader(FILES[label]).get("events"))

        for i, frame in enumerate(events):
            if not mask[i]: continue
            n_events_mask += 1
            for cname in barrel_colls:
                try:
                    for h in frame.get(cname):
                        pos = h.getPosition()
                        r   = math.sqrt(pos.x**2 + pos.y**2)
                        layer_idx = int((r - ECAL_INNER_R) / ECAL_LAYER_DZ)
                        if 0 <= layer_idx < N_LAYERS:
                            layer_edep[layer_idx] += h.getEnergy()
                except Exception:
                    pass

        if n_events_mask == 0: continue
        mean_profile = layer_edep / n_events_mask
        ax.plot(np.arange(N_LAYERS), mean_profile,
                color=COLORS[label], linewidth=2,
                label=f"{label}  (N={n_events_mask})")

    ax.set_xlabel("ECAL barrel layer index  (0 = innermost, ~1843 mm)")
    ax.set_ylabel("Mean E_dep per layer per event [GeV]")
    ax.set_title(
        "Longitudinal shower profile — mean energy deposit vs ECAL barrel layer\n"
        "Physics question: does CC3 place the shower maximum at the right depth?\n"
        "Barrel only | layer from r=√(x²+y²), inner face 1843 mm, pitch 5.4 mm | 1–10 GeV gun",
        fontsize=9
    )
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "07_longitudinal_profile.png")


# ─────────────────────────────────────────────
#  Plot 8 — Transverse shower profile
#  Physics question: does CC3 reproduce the lateral shower spread (Molière radius)?
#  Gun direction is (1,0,0) so shower axis = x-axis.
#  Transverse distance of each hit: r_perp = sqrt(y² + z²).
#  Molière radius in W/Si ≈ 9 mm → 5 mm bins give ~2 bins per Molière radius.
#  Log y-axis: shower core is ~100× brighter than tails — log shows both clearly.
# ─────────────────────────────────────────────
def plot_transverse_profile(data):
    import podio.root_io as rio
    fig, ax = plt.subplots(figsize=(9, 5))

    r_edges   = np.arange(0, 105, 5)        # 0–100 mm, 5 mm bins
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    for label, d in data.items():
        if d is None: continue
        mask          = get_mask(label, d)
        radial_edep   = np.zeros(len(r_centers))
        n_events_mask = 0
        events        = list(rio.Reader(FILES[label]).get("events"))

        for i, frame in enumerate(events):
            if not mask[i]: continue
            n_events_mask += 1
            for cname in ECAL_COLLS:
                try:
                    for h in frame.get(cname):
                        pos   = h.getPosition()
                        # shower axis = x-axis, transverse plane is y-z
                        r_perp    = math.sqrt(pos.y**2 + pos.z**2)
                        bin_idx   = int(r_perp / 5.0)
                        if 0 <= bin_idx < len(r_centers):
                            radial_edep[bin_idx] += h.getEnergy()
                except Exception:
                    pass

        if n_events_mask == 0: continue
        mean_profile = radial_edep / n_events_mask
        ax.plot(r_centers, mean_profile,
                color=COLORS[label], linewidth=2,
                label=f"{label}  (N={n_events_mask})")

    ax.set_xlabel("Transverse distance from shower axis  r_⊥ = √(y²+z²) [mm]")
    ax.set_ylabel("Mean E_dep per radial bin per event [GeV]")
    ax.set_yscale("log")
    ax.set_title(
        "Transverse shower profile — mean energy deposit vs distance from shower axis\n"
        "Physics question: does CC3 reproduce the lateral shower spread (Molière radius)?\n"
        "Shower axis = x-axis (gun dir=(1,0,0)) | Molière radius W/Si ≈ 9 mm | bin = 5 mm",
        fontsize=9
    )
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "08_transverse_profile.png")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  CC3 vs G4 photon shower analysis — v4")
    print("  ECal: barrel + endcap collections")
    print("  New plots: z0/z20 comparison, longitudinal, transverse profiles")
    print("=" * 65)

    data    = {label: process_file(label, path) for label, path in FILES.items()}
    data_z0 = {label: process_file(label, path) for label, path in FILES_Z0.items()}

    print("\n\nGenerating plots →", OUT_DIR)
    plot_accounting(data)
    plot_edep_spectrum(data)
    plot_energy_response(data)
    plot_hit_multiplicity(data)
    plot_hits_vs_energy(data)
    plot_z0_vs_z20(data, data_z0)
    plot_longitudinal_profile(data)
    plot_transverse_profile(data)

    print("\n[DONE]")
