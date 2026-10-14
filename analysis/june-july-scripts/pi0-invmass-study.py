"""
pi0_invariant_mass.py
=====================
π⁰ → γγ invariant mass analysis: MC truth + reco (Tier 1 + Tier 2).

Physics questions:
  Analysis A (truth): Sharp peak at 134.97 MeV? G4 == CC3? Sanity check.
  Tier 1 (reco):      Full di-photon spectrum, no phase-space cuts.
                      Honest deployed-chain comparison.
  Tier 2 (reco):      E_mc > 10 GeV on both matched photons.
                      CC3-active phase space, same population in both files.

Link strategy: Strategy A — highest-weight match per reco PFO.
Signal (reco): PandoraPFO type==22, matched MC genStatus==1 PDG==22, weight>0.5.
Tier 2 gate  : E_mc > 10 GeV on BOTH matched MC photons in the pair.

Run:
    source ~/source.sh
    python3 pi0_invariant_mass.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from podio import root_io
from collections import defaultdict
import os, math

# ── CONFIG ────────────────────────────────────────────────────────────────────
USE_1K = False   # flip to False for 10k run

if USE_1K:
    G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root"
    CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root"
    MAX_EVENTS = 1000
    RUN_LABEL  = "1k events [dev]"
else:
    G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root"
    CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"
    MAX_EVENTS = 10000
    RUN_LABEL  = "10k events"

PLOT_DIR = os.path.expanduser(
    f"~/thesis-ml-sim/plots/pi0_invariant_mass_{'1k' if USE_1K else '10k'}_v1"
)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── PHYSICS ───────────────────────────────────────────────────────────────────
PHOTON_PDG    = 22
PI0_PDG       = 111
E_MC_TIER2    = 10.0   # GeV — CC3 active phase space gate (applied to E_mc)
WEIGHT_CUT    = 0.5
PI0_MASS_MEV  = 134.97 # MeV — expected truth peak position

# ── HISTOGRAM SETTINGS ────────────────────────────────────────────────────────
# Range: 80-200 MeV. Bin width: 2 MeV = 60 bins.
# Motivated by ILD ECAL stochastic term ~15%/sqrt(E):
# at 15 GeV: sigma_E/E ~ 3.9%, sigma_m ~ 10-20 MeV → need ~5 bins across peak.
M_BINS     = np.arange(0.080, 0.201, 0.002)   # GeV
M_CENTERS  = 0.5 * (M_BINS[:-1] + M_BINS[1:])
M_BINS_TRUTH = np.arange(0.120, 0.155, 0.001) # tighter range for truth (sub-MeV check)
M_CENTERS_TRUTH = 0.5 * (M_BINS_TRUTH[:-1] + M_BINS_TRUTH[1:])

# ── COLORS / STYLE ────────────────────────────────────────────────────────────
C_G4  = "#2166ac"
C_CC3 = "#d6604d"

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

# ── INVARIANT MASS ────────────────────────────────────────────────────────────
def m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2):
    """
    4-vector invariant mass. Guard against negative m^2 from numerical precision
    on nearly collinear photons.
    """
    m2 = 2.0 * (e1*e2 - px1*px2 - py1*py2 - pz1*pz2)
    return math.sqrt(max(m2, 0.0))

# ── FIT MODEL ─────────────────────────────────────────────────────────────────
def gauss_plus_poly(x, N, mu, sigma, a0, a1, a2):
    return N * np.exp(-0.5 * ((x - mu) / sigma)**2) + a0 + a1*x + a2*x**2

# ── TRUTH-LEVEL ANALYSIS (Analysis A) ────────────────────────────────────────
def run_truth(filepath, label):
    """
    Strategy: reverse walk — for each genStatus==1 photon, check if parent
    is a π⁰. Group photons by parent object ID. For each π⁰ parent with
    exactly 2 photon daughters in the group, compute M_γγ.

     π⁰ generatorStatus is 2 in this sample (Pythia intermediate decay product).
     We filter on the PHOTON's genStatus==1, not the π⁰'s — correct.
     Confirmed by diagnostic: both MCParticles and MCParticlesSkimmed give
     identical results for signal photons. Using MCParticles for consistency.
    """
    print(f"\n{'='*55}")
    print(f"  Analysis A (truth): {label}  [{RUN_LABEL}]")
    print(f"{'='*55}")

    reader = root_io.Reader(filepath)
    masses = []
    n_events = 0
    n_pi0_found = 0

    # Probe first event to report what genStatus values pi0s carry
    probed = False

    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        # Group genStatus==1 photons by their pi0 parent's object ID
        pi0_photons = defaultdict(list)  # pi0_obj_id -> list of (e, px, py, pz)

        for mc in event.get("MCParticles"):
            if mc.getGeneratorStatus() != 1:
                continue
            if mc.getPDG() != PHOTON_PDG:
                continue

            parents = mc.getParents()
            if len(parents) == 0:
                continue
            parent = parents[0]

            # Probe: print pi0 genStatus in first event
            if not probed and parent.getPDG() == PI0_PDG:
                print(f"  [probe] First π⁰ parent seen: "
                      f"PDG={parent.getPDG()}, "
                      f"genStatus={parent.getGeneratorStatus()}, "
                      f"E={parent.getEnergy():.2f} GeV")
                probed = True

            if parent.getPDG() != PI0_PDG:
                continue

            mom = mc.getMomentum()
            e   = mc.getEnergy()
            pi0_photons[parent.getObjectID().index].append(
                (e, mom.x, mom.y, mom.z)
            )

        # For each π⁰ with exactly 2 photon daughters: compute M_γγ
        for pi0_id, photons in pi0_photons.items():
            if len(photons) != 2:
                continue   # skip if not exactly 2 (rare: split decay, secondaries)
            n_pi0_found += 1
            e1, px1, py1, pz1 = photons[0]
            e2, px2, py2, pz2 = photons[1]
            masses.append(m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2))

    masses = np.array(masses)
    print(f"  Events processed : {n_events}")
    print(f"  π⁰ pairs found   : {n_pi0_found}")
    print(f"  M_γγ computed    : {len(masses)}")
    if len(masses) > 0:
        print(f"  Mean M_γγ [MeV]  : {1000*np.mean(masses):.2f}")
        print(f"  Std  M_γγ [MeV]  : {1000*np.std(masses):.2f}  "
              f"← should be < 1 MeV if correct")
    return masses


# ── RECO-LEVEL ANALYSIS ───────────────────────────────────────────────────────
def run_reco(filepath, label):
    """
    Collects all photon PFO pairs per event.
    Returns two lists of M_γγ values:
      tier1: all pairs (no energy gate)
      tier2: pairs where BOTH matched MC photons have E_mc > 10 GeV

    Link strategy: highest-weight match per reco PFO (Strategy A).
    Match requirement: genStatus==1, PDG==22, weight > 0.5.
    """
    print(f"\n{'='*55}")
    print(f"  Reco analysis: {label}  [{RUN_LABEL}]")
    print(f"{'='*55}")

    reader = root_io.Reader(filepath)
    tier1_masses = []
    tier2_masses = []
    n_events = 0
    n_pairs_t1 = 0
    n_pairs_t2 = 0

    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        # Strategy A: best-weight MC match per reco PFO
        reco_to_mc   = {}
        reco_to_wgt  = {}
        for link in event.get("RecoMCTruthLink"):
            reco   = link.getFrom()
            mc     = link.getTo()
            weight = link.getWeight()
            rid    = reco.getObjectID().index
            if rid not in reco_to_wgt or weight > reco_to_wgt[rid]:
                reco_to_mc[rid]  = mc
                reco_to_wgt[rid] = weight

        # Collect selected photon PFOs: type==22, matched to genStatus==1 photon, weight>0.5
        photons = []  # each entry: (e, px, py, pz, e_mc)
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue
            rid = pfo.getObjectID().index
            if rid not in reco_to_mc:
                continue
            if reco_to_wgt[rid] <= WEIGHT_CUT:
                continue
            mc = reco_to_mc[rid]
            if mc.getGeneratorStatus() != 1:
                continue
            if mc.getPDG() != PHOTON_PDG:
                continue

            e_mc = mc.getEnergy()
            mom  = pfo.getMomentum()
            e    = pfo.getEnergy()
            photons.append((e, mom.x, mom.y, mom.z, e_mc))

        # Form all pairs
        n = len(photons)
        for i in range(n):
            for j in range(i+1, n):
                e1, px1, py1, pz1, emc1 = photons[i]
                e2, px2, py2, pz2, emc2 = photons[j]
                mass = m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2)

                # Tier 1: all pairs
                tier1_masses.append(mass)
                n_pairs_t1 += 1

                # Tier 2: E_mc > 10 GeV on BOTH matched MC photons
                if emc1 > E_MC_TIER2 and emc2 > E_MC_TIER2:
                    tier2_masses.append(mass)
                    n_pairs_t2 += 1

    print(f"  Events processed : {n_events}")
    print(f"  Tier 1 pairs     : {n_pairs_t1}  (all photon pairs)")
    print(f"  Tier 2 pairs     : {n_pairs_t2}  (E_mc > {E_MC_TIER2} GeV both photons)")
    return np.array(tier1_masses), np.array(tier2_masses)


# ── FIT HELPER ────────────────────────────────────────────────────────────────
def fit_gaussian(centers, counts, label):
    """
    Gaussian + quadratic background fit.
    Constrained: mu in [100,200] MeV, sigma in [5,40] MeV, N > 0.
    Fit window: 100-200 MeV only (signal region).
    Starting guess: mu = 140 MeV (physics prior).
    """
    # Fit only the signal window — ignore far sidebands
    fit_mask = (centers > 0.100) & (centers < 0.200)
    x = centers[fit_mask]
    y = counts[fit_mask]

    if len(x) < 6:
        print(f"  Fit [{label}] SKIPPED: too few bins in fit window")
        return None, None

    # Starting guesses — mu anchored to PDG value
    mu0    = 0.140
    sigma0 = 0.015
    N0     = y.max() * sigma0 * np.sqrt(2 * np.pi)
    p0     = [N0, mu0, sigma0, 1.0, 0.0, 0.0]

    # Bounds: (lower, upper) for each parameter
    # [  N,    mu,    sigma,   a0,    a1,    a2  ]
    lower = [  0,  0.100,  0.005, -1e6, -1e6, -1e6]
    upper = [1e9,  0.200,  0.040,  1e6,  1e6,  1e6]

    try:
        popt, pcov = curve_fit(gauss_plus_poly, x, y,
                               p0=p0, bounds=(lower, upper),
                               maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        N, mu, sigma = popt[0], popt[1], popt[2]
        dN, dmu, dsigma = perr[0], perr[1], perr[2]
        print(f"  Fit [{label}]: μ = {1000*mu:.2f} ± {1000*dmu:.2f} MeV | "
              f"σ = {1000*sigma:.2f} ± {1000*dsigma:.2f} MeV | "
              f"N = {abs(N):.0f} ± {dN:.0f}")
        return popt, perr
    except RuntimeError as e:
        print(f"  Fit [{label}] FAILED: {e}")
        return None, None# ── PLOT: TRUTH M_γγ ──────────────────────────────────────────────────────────
def plot_truth(g4_masses, cc3_masses):
    """
    Physics question: do G4 and CC3 give identical truth M_γγ?
    Expected: sharp peak at 134.97 MeV, width < 1 MeV, no background.
    Any difference between G4 and CC3 here = bug in the analysis code.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(1000*g4_masses,  bins=1000*M_BINS_TRUTH, histtype="step",
            color=C_G4,  linewidth=2, label=f"Geant4  (N={len(g4_masses)})")
    ax.hist(1000*cc3_masses, bins=1000*M_BINS_TRUTH, histtype="step",
            color=C_CC3, linewidth=2, linestyle="--",
            label=f"CC3  (N={len(cc3_masses)})")
    ax.axvline(PI0_MASS_MEV, color="black", lw=1.0, ls=":", alpha=0.7,
               label=f"PDG π⁰ mass ({PI0_MASS_MEV} MeV)")

    ax.set_xlabel("M$_{\\gamma\\gamma}$ [MeV]")
    ax.set_ylabel("π⁰ candidates / bin")
    ax.set_title(
        "Analysis A — MC Truth M$_{\\gamma\\gamma}$: G4 vs CC3\n"
        "genStatus==1 photons, π⁰ parent, exactly 2 daughters per π⁰\n"
        "No detector effects. Width > 1 MeV = bug in pairing logic.",
        fontsize=10
    )
    ax.legend(framealpha=0.5)
    ax.text(0.01, 1.01, f"ILD sim — {RUN_LABEL} — raw counts",
            transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    out = f"{PLOT_DIR}/analysisA_truth_mgg.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓  Saved: analysisA_truth_mgg.png")


# ── PLOT: RECO M_γγ (TIER 1 + TIER 2 SIDE BY SIDE) ──────────────────────────
def plot_reco(g4_t1, cc3_t1, g4_t2, cc3_t2, tier_label, filename, fit=True):
    """
    Physics question:
      Tier 1: Does CC3 reproduce the full reco di-photon spectrum?
      Tier 2: In the CC3-active phase space (E_mc>10 GeV), does CC3
              reproduce the peak position, width, and yield?

    Notes on normalization:
      Raw counts shown — event counts are equal (same N events both files).
      If pair yields differ by >5%, state explicitly (more PFOs = more pairs).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, g4_m, cc3_m, tier, subtitle in [
        (axes[0], g4_t1, cc3_t1, "Tier 1",
         "All reco photon pairs\n(no energy gate, full chain)"),
        (axes[1], g4_t2, cc3_t2, "Tier 2",
         "E$_{mc}$ > 10 GeV on both matched photons\n(CC3-active phase space)"),
    ]:
        g4_counts,  _ = np.histogram(g4_m,  bins=M_BINS)
        cc3_counts, _ = np.histogram(cc3_m, bins=M_BINS)

        ax.hist(g4_m,  bins=M_BINS, histtype="step",
                color=C_G4,  linewidth=2,
                label=f"Geant4  (N={len(g4_m)})")
        ax.hist(cc3_m, bins=M_BINS, histtype="step",
                color=C_CC3, linewidth=2, linestyle="--",
                label=f"CC3  (N={len(cc3_m)})")

        ax.axvline(PI0_MASS_MEV/1000, color="gray", lw=1.0, ls=":",
                   alpha=0.7, label=f"PDG mass ({PI0_MASS_MEV:.0f} MeV)")

        # Fit Gaussians if enough statistics
        if fit and len(g4_m) > 30:
            print(f"\n  Fitting {tier}:")
            for counts, label, color in [
                (g4_counts,  "G4",  C_G4),
                (cc3_counts, "CC3", C_CC3),
            ]:
                popt, perr = fit_gaussian(M_CENTERS, counts, label)
                if popt is not None:
                    x_fit = np.linspace(M_BINS[0], M_BINS[-1], 500)
                    ax.plot(x_fit, gauss_plus_poly(x_fit, *popt),
                            color=color, lw=1.5, ls="-", alpha=0.7)

        ax.set_xlabel("M$_{\\gamma\\gamma}$ [GeV]")
        ax.set_ylabel("Pairs / 2 MeV")
        ax.set_title(f"Reco M$_{{\\gamma\\gamma}}$ — {tier}\n{subtitle}", fontsize=10)
        ax.legend(framealpha=0.5, fontsize=9)

        # Note on split-off spike for Tier 1
        if tier == "Tier 1":
            ax.annotate("← split-off spike\n(PandoraPFA cluster splitting)",
                        xy=(0.085, 0), xytext=(0.100, max(g4_counts)*0.4),
                        fontsize=7, color="gray",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    axes[0].text(0.01, 1.01,
                 f"ILD reco — {RUN_LABEL} — genStatus==1, weight>0.5 — raw counts",
                 transform=axes[0].transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    out = f"{PLOT_DIR}/{filename}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓  Saved: {filename}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  pi0_invariant_mass.py  v1")
print("  Analysis A (truth) + Tier 1 + Tier 2 reco M_γγ")
print("="*55)

print("\n── Analysis A: MC Truth ──")
g4_truth  = run_truth(G4_FILE,  "Geant4")
cc3_truth = run_truth(CC3_FILE, "CC3")

print("\n── Reco: G4 ──")
g4_t1, g4_t2 = run_reco(G4_FILE, "Geant4")

print("\n── Reco: CC3 ──")
cc3_t1, cc3_t2 = run_reco(CC3_FILE, "CC3")

print("\n── Generating plots ──")
plot_truth(g4_truth, cc3_truth)
plot_reco(g4_t1, cc3_t1, g4_t2, cc3_t2,
          tier_label="Tier1+Tier2",
          filename="reco_mgg_tier1_tier2.png")

print(f"\n  All plots saved to: {PLOT_DIR}")
print(f"  Done.\n")

