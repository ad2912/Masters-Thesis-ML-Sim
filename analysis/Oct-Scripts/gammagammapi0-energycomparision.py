#!/usr/bin/env python3
"""
plot_gammagammapi0_energy_compare.py
Second plotting script for the GammaGammaCandidatePi0s .npz files (G4 vs CC3). Floor 0.5, mass formula A.

Plots
  10  G4 two_distinct scaled behind the three CC3 simulation-split panels.
      Panels both_g4 and one_each: G4 = all two_distinct.
      Panel both_fastsim: G4 = two_distinct with BOTH linked true photons E >= 10 GeV.
      G4 is scaled to the same number of entries as the CC3 panel (shape comparison only).
  11  Median mass vs pair-energy group (same_pi0 pairs), G4 vs CC3.
  12  Normalised mass shapes in each pair-energy group (same_pi0 pairs), G4 vs CC3.
  13  CC3 / G4 ratio: raw yield and area-normalised shape, for two_distinct and same_pi0 (3 MeV bins).
Pair energy = energy of the two reco photons added together (the energy of the reconstructed pi0).
Run: python3 plot_gammagammapi0_energy_compare.py   (env overrides: G4_NPZ, CC3_NPZ, PLOT_OUT)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ = {
    "G4":  os.environ.get("G4_NPZ",  "/data/dust/user/alimuham/thesis/reco/gammagammapi0-candidates-g4.npz"),
    "CC3": os.environ.get("CC3_NPZ", "/data/dust/user/alimuham/thesis/reco/gammagammapi0-candidates-cc3.npz"),
}
OUT = os.environ.get("PLOT_OUT", "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/gammagammapi0-candidates-energy-comparision-100k")
FLOOR = 0.5
EDGES = [5, 10, 20, 30, 50, 80, 125]          # pair-energy group edges [GeV]
LABELS = ["G4", "CC3"]
COLOR = {"G4": "#2166ac", "CC3": "#d6604d"}
NAME = {"G4": "Geant4", "CC3": "CC3"}

PI0_MASS = 0.1349766
FINDER_WINDOW = (PI0_MASS - 0.04, PI0_MASS + 0.04)
BINS = np.linspace(0.090, 0.180, 46)          # 2 MeV
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])
BINS3 = np.linspace(0.090, 0.180, 31)         # 3 MeV, for ratio plots
CENTERS3 = 0.5 * (BINS3[:-1] + BINS3[1:])

WEAK, PHOTON, OTHER = 0, 1, 2
SIM_SPLITS = ["both_g4", "one_each", "both_fastsim"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 13, "axes.titlesize": 12, "axes.labelsize": 14,
    "legend.fontsize": 10, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
LOG = []
def log(s=""):
    print(s, flush=True); LOG.append(s)

# ------------------------------------------------------------------ loading
NEED = []
for k in (0, 1):
    NEED += [f"d{k}_{c}" for c in ("E", "px", "py", "pz")]
    NEED += [f"d{k}_l0_{c}" for c in ("cw", "pdg", "idx", "fast", "pi0_par_idx", "E")]

def load(path):
    z = np.load(path, allow_pickle=False)
    d = {c: z[c] for c in NEED}
    d["n"] = len(d["d0_E"])
    d["events_read"] = int(z["meta_events_read"])
    return d

# ------------------------------------------------------------------ same definitions as script 1
def daughter_state(d, k, floor):
    cw, pdg = d[f"d{k}_l0_cw"], d[f"d{k}_l0_pdg"]
    s = np.full(len(cw), OTHER, dtype=np.int8)
    s[pdg == 22] = PHOTON
    s[cw < floor] = WEAK
    return s

def pair_class(d, floor):
    s0, s1 = daughter_state(d, 0, floor), daughter_state(d, 1, floor)
    n_ph = (s0 == PHOTON).astype(int) + (s1 == PHOTON).astype(int)
    same_mc = d["d0_l0_idx"] == d["d1_l0_idx"]
    cls = np.full(d["n"], -1, dtype=np.int8)
    cls[n_ph == 0] = 0; cls[n_ph == 1] = 1
    cls[(n_ph == 2) & same_mc] = 2; cls[(n_ph == 2) & ~same_mc] = 3
    assert (cls >= 0).all()
    return cls

def sim_split_masks(d):
    f0, f1 = d["d0_l0_fast"] == 1, d["d1_l0_fast"] == 1
    return {"both_g4": (~f0) & (~f1), "one_each": f0 ^ f1, "both_fastsim": f0 & f1}

def same_pi0_mask(d):
    p0, p1 = d["d0_l0_pi0_par_idx"], d["d1_l0_pi0_par_idx"]
    return (p0 >= 0) & (p1 >= 0) & (p0 == p1)

def masses(d):
    E = d["d0_E"] + d["d1_E"]
    px = d["d0_px"] + d["d1_px"]; py = d["d0_py"] + d["d1_py"]; pz = d["d0_pz"] + d["d1_pz"]
    mA = np.sqrt(np.clip(E**2 - px**2 - py**2 - pz**2, 0, None))
    dot = d["d0_px"]*d["d1_px"] + d["d0_py"]*d["d1_py"] + d["d0_pz"]*d["d1_pz"]
    mB = np.sqrt(np.clip(2.0 * (d["d0_E"]*d["d1_E"] - dot), 0, None))
    return mA, mB

# ------------------------------------------------------------------ new helpers
def energy_group(pair_E, edges):
    """group number 0..len(edges)-2 for edges[g] <= E < edges[g+1]; -1 if outside the range"""
    g = np.digitize(pair_E, edges) - 1
    g[(pair_E < edges[0]) | (pair_E >= edges[-1])] = -1
    return g

def median_and_err(m):
    if len(m) < 2:
        return np.nan, np.nan
    return np.median(m), 1.2533 * m.std() / np.sqrt(len(m))

def draw(ax, m, label, color, ls="-", bins=BINS):
    c, _ = np.histogram(m, bins=bins)
    ax.stairs(c, bins, color=color, linewidth=2.0, linestyle=ls, label=label)
    ax.errorbar(0.5 * (bins[:-1] + bins[1:]), c, yerr=np.sqrt(c), fmt="none", ecolor=color, elinewidth=0.9)

def edges_lines(ax):
    ax.axvline(PI0_MASS, color="gray", lw=0.9, ls=":")
    for e in FINDER_WINDOW:
        ax.axvline(e, color="gray", lw=0.7, ls="--", alpha=0.6)

def footer(fig, extra=""):
    txt = (f"Inputs: {os.path.basename(NPZ['G4'])} (G4), {os.path.basename(NPZ['CC3'])} (CC3). "
           f"Link weight floor {FLOOR}. Mass formula A. Error bars sqrt(N). Gray dashed = finder window, dotted = 0.135 GeV.")
    fig.text(0.5, -0.01, txt + ("\n" + extra if extra else ""), ha="center", va="top", fontsize=9)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig); log(f"  saved {name}")

def group_name(i):
    return f"{EDGES[i]}-{EDGES[i+1]} GeV"

# ------------------------------------------------------------------ plot 10
def plot_overlay(D, M, CLS):
    td_cc3 = CLS["CC3"] == 3
    td_g4 = CLS["G4"] == 3
    e10_g4 = (D["G4"]["d0_l0_E"] >= 10) & (D["G4"]["d1_l0_E"] >= 10)
    masks = sim_split_masks(D["CC3"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    log("\n[plot 10] G4 overlay scaling")
    for ax, sname in zip(axes, SIM_SPLITS):
        m_cc3 = M["CC3"][td_cc3 & masks[sname]]
        use_e10 = sname == "both_fastsim"
        m_g4 = M["G4"][td_g4 & e10_g4] if use_e10 else M["G4"][td_g4]
        scale = len(m_cc3) / max(len(m_g4), 1)
        cg, _ = np.histogram(m_g4, bins=BINS)
        glab = (f"G4 two_distinct{', both true E>=10 GeV' if use_e10 else ''} "
                f"(N={len(m_g4):,}, scaled x{scale:.3f})")
        ax.stairs(cg * scale, BINS, fill=True, color=COLOR["G4"], alpha=0.25, label=glab)
        ax.stairs(cg * scale, BINS, color=COLOR["G4"], linewidth=1.2)
        draw(ax, m_cc3, f"CC3 {sname} (N={len(m_cc3):,})", COLOR["CC3"])
        edges_lines(ax)
        ax.set_xlim(BINS[0], BINS[-1]); ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$M_{\gamma\gamma}$ [GeV]"); ax.set_ylabel("candidates / 2 MeV")
        ax.set_title(f"two_distinct, {sname}")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        log(f"  {sname}: CC3 N={len(m_cc3)}, G4 reference N={len(m_g4)}, scale={scale:.4f}, "
            f"median CC3 {1000*np.median(m_cc3):.2f} vs G4 {1000*np.median(m_g4):.2f} MeV")
    fig.suptitle("Does each CC3 sub-class have the same shape as G4?  (G4 scaled to the CC3 entries; shape only)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    footer(fig, "E>=10 GeV on G4 matches only the both_fastsim panel; theta/phi acceptance is NOT matched.")
    save(fig, "10_overlay_g4_behind_cc3_simulation_split.png")

# ------------------------------------------------------------------ plots 11, 12
def plot_energy(D, M, CLS):
    ng = len(EDGES) - 1
    G, SEL = {}, {}
    for lab in LABELS:
        d = D[lab]
        SEL[lab] = (CLS[lab] == 3) & same_pi0_mask(d)
        pair_E = d["d0_E"] + d["d1_E"]
        G[lab] = energy_group(pair_E, EDGES)
        out = int((SEL[lab] & (G[lab] < 0)).sum())
        log(f"  {lab}: same_pi0 pairs {int(SEL[lab].sum()):,}; outside {EDGES[0]}-{EDGES[-1]} GeV: {out:,}")
    log("\n[plot 11] median mass per pair-energy group, same_pi0 pairs (error = 1.2533*std/sqrt(N), independent samples assumed)")
    log(f"  {'group':>12s} {'N G4':>8s} {'N CC3':>8s} {'median G4':>16s} {'median CC3':>16s} {'CC3-G4 [MeV]':>16s}")
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for lab, off in (("G4", 0.98), ("CC3", 1.02)):
        xs, ys, es, xe = [], [], [], []
        for i in range(ng):
            m = M[lab][SEL[lab] & (G[lab] == i)]
            med, err = median_and_err(m)
            xs.append(np.sqrt(EDGES[i] * EDGES[i+1]) * off); ys.append(med); es.append(err)
            xe.append([np.sqrt(EDGES[i]*EDGES[i+1]) - EDGES[i], EDGES[i+1] - np.sqrt(EDGES[i]*EDGES[i+1])])
        ax.errorbar(xs, ys, yerr=es, xerr=np.array(xe).T * 0.9, fmt="o", color=COLOR[lab], capsize=3,
                    label=NAME[lab], ls="none")
    for i in range(ng):
        r = []
        for lab in LABELS:
            m = M[lab][SEL[lab] & (G[lab] == i)]
            r.append((len(m),) + median_and_err(m))
        diff = r[1][1] - r[0][1]; derr = np.hypot(r[0][2], r[1][2])
        log(f"  {group_name(i):>12s} {r[0][0]:8d} {r[1][0]:8d} "
            f"{1000*r[0][1]:8.2f}+-{1000*r[0][2]:5.2f} {1000*r[1][1]:8.2f}+-{1000*r[1][2]:5.2f} "
            f"{1000*diff:+8.2f}+-{1000*derr:5.2f}")
    ax.axhline(PI0_MASS, color="gray", ls=":", label="0.135 GeV")
    ax.set_xscale("log"); ax.set_xticks(EDGES); ax.set_xticklabels([str(e) for e in EDGES])
    ax.set_xlabel("pair energy (two reco photons added) [GeV]")
    ax.set_ylabel(r"median $M_{\gamma\gamma}$ [GeV]")
    ax.set_title("Does the reco mass depend on the pi0 energy?  same_pi0 pairs")
    ax.legend(frameon=False)
    footer(fig, "Points at group centres (geometric mean), horizontal bars = group width. Median error 1.2533*std/sqrt(N).")
    save(fig, "11_median_mass_vs_pair_energy.png")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for i, ax in enumerate(axes.ravel()):
        if i >= ng:
            ax.axis("off"); continue
        for lab in LABELS:
            m = M[lab][SEL[lab] & (G[lab] == i)]
            c, _ = np.histogram(m, bins=BINS)
            n = max(len(m), 1)
            ax.stairs(c / n, BINS, color=COLOR[lab], linewidth=2, linestyle="-" if lab == "G4" else "--",
                      label=f"{NAME[lab]} (N={len(m):,})")
            ax.errorbar(CENTERS, c / n, yerr=np.sqrt(c) / n, fmt="none", ecolor=COLOR[lab], elinewidth=0.8)
        edges_lines(ax)
        ax.set_xlim(BINS[0], BINS[-1]); ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$M_{\gamma\gamma}$ [GeV]"); ax.set_ylabel("fraction of pairs in group / 2 MeV")
        ax.set_title(f"pair energy {group_name(i)}")
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Mass shape inside each pair-energy group (each curve scaled to unit area).  same_pi0 pairs",
                 fontsize=14, y=1.0)
    plt.tight_layout(); footer(fig)
    save(fig, "12_mass_shapes_by_pair_energy.png")

# ------------------------------------------------------------------ plot 13
def plot_ratios(D, M, CLS):
    rows = [("two_distinct", lambda lab: CLS[lab] == 3),
            ("same_pi0", lambda lab: (CLS[lab] == 3) & same_pi0_mask(D[lab]))]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    log("\n[plot 13] CC3 / G4 ratios (3 MeV bins)")
    for r, (name, fsel) in enumerate(rows):
        a, _ = np.histogram(M["G4"][fsel("G4")], bins=BINS3)
        b, _ = np.histogram(M["CC3"][fsel("CC3")], bins=BINS3)
        Na, Nb = a.sum(), b.sum()
        ok = (a > 0) & (b > 0)
        ratio = np.where(ok, b / np.where(a > 0, a, 1), np.nan)
        err = ratio * np.sqrt(np.where(ok, 1.0 / np.where(a > 0, a, 1) + 1.0 / np.where(b > 0, b, 1), np.nan))
        shape = ratio * Na / Nb
        for c, (y, yl, ttl, ref) in enumerate((
                (ratio, "CC3 / G4 (raw counts)", "raw yield", Nb / Na),
                (shape, "CC3 / G4 (each scaled to unit area)", "shape only", 1.0))):
            ax = axes[r][c]
            ax.errorbar(CENTERS3, y, yerr=err * (1 if c == 0 else Na / Nb), fmt="o", ms=4, color="black", ecolor="gray")
            ax.axhline(1.0, color="gray", lw=1.0)
            if c == 0:
                ax.axhline(ref, color=COLOR["CC3"], lw=1.2, ls="--", label=f"overall yield ratio {ref:.3f}")
                ax.legend(frameon=False, loc="upper left")
            edges_lines(ax)
            ax.set_xlim(BINS3[0], BINS3[-1]); ax.set_ylim(0.5, 1.5)
            ax.set_xlabel(r"$M_{\gamma\gamma}$ [GeV]"); ax.set_ylabel(yl)
            ax.set_title(f"{name}: {ttl}  (G4 N={Na:,}, CC3 N={Nb:,})")
        log(f"  {name}: G4 N={Na}, CC3 N={Nb}, overall yield ratio {Nb/Na:.4f}")
    fig.suptitle("Where does CC3 differ from G4, in yield and in shape?", fontsize=14, y=1.0)
    plt.tight_layout()
    footer(fig, "Ratio error from Poisson counts. G4 and CC3 share generator events, so errors are approximate.")
    save(fig, "13_ratio_cc3_over_g4.png")

# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    D = {lab: load(NPZ[lab]) for lab in LABELS}
    M, CLS, MB = {}, {}, {}
    for lab in LABELS:
        M[lab], MB[lab] = masses(D[lab])
        CLS[lab] = pair_class(D[lab], FLOOR)
        diff = 1000 * np.abs(M[lab] - MB[lab])
        log(f"{lab}: N={D[lab]['n']:,}  |M_A - M_B|: median {np.median(diff):.4f} MeV, 99th pct {np.percentile(diff, 99):.3f}, max {diff.max():.2f} MeV")
    e10_cc3 = (D["CC3"]["d0_l0_E"] >= 10) & (D["CC3"]["d1_l0_E"] >= 10) & (CLS["CC3"] == 3)
    log(f"CC3 two_distinct with both true E>=10 GeV: {int(e10_cc3.sum()):,}; of these both_fastsim: "
        f"{int((e10_cc3 & sim_split_masks(D['CC3'])['both_fastsim']).sum()):,}; "
        f"all both_fastsim: {int(((CLS['CC3']==3) & sim_split_masks(D['CC3'])['both_fastsim']).sum()):,}")
    plot_overlay(D, M, CLS)
    plot_energy(D, M, CLS)
    plot_ratios(D, M, CLS)
    with open(os.path.join(OUT, "summary_energy_compare.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    log(f"\nSaved to {OUT}")

if __name__ == "__main__":
    main()
