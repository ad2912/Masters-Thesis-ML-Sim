#!/usr/bin/env python3
"""
plot_gammagammapi0_candidates.py
Pi0 -> gamma gamma candidates from GammaGammaCandidatePi0s, G4 vs CC3, from the two .npz files.

Run:  python3 plot_gammagammapi0_candidates.py
(Optional overrides via env vars: G4_NPZ, CC3_NPZ, PLOT_OUT)

Vocabulary
  candidate = one pair (one row) = two reco photon PFOs, called daughters d0 (higher E) and d1.
  top link  = the MC particle with the highest cluster weight for that daughter (columns dK_l0_*).
Daughter state (per daughter, at weight floor f), checked in this order:
  weak   : top-link cluster weight < f   (PDG is NOT looked at)
  photon : weight >= f and top-link PDG == 22
  other  : weight >= f and top-link PDG != 22   (e+- are in here)
Pair class (depends on how many daughters are 'photon'):
  no_photon    : 0
  one_photon   : 1
  two_same     : 2, both top links point to the SAME MC photon (same MC index)  -> fragments
  two_distinct : 2, top links point to DIFFERENT MC photons                       -> clean (link sense)
Splits inside two_distinct only:
  simulation (flag of each daughter's top-link MC photon): both_g4 / one_each / both_fastsim
  parentage  (any parent of the MC photon has PDG 111 = 'pi0 parent'):
             same_pi0 / different_pi0s / one_from_pi0 / none_from_pi0
Mass formula: A = full  sqrt(E^2 - |p|^2)  of the summed daughters (default)
              B = massless  sqrt(2 (E1E2 - p1.p2))   (diagnostic only)
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
OUT = os.environ.get("PLOT_OUT", "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/gammagammapi0-candidates-V1-100k")
FLOORS = [0.5, 0.1]
PRIMARY_FLOOR = 0.5
LABELS = ["G4", "CC3"]
COLOR = {"G4": "#2166ac", "CC3": "#d6604d"}
NAME = {"G4": "Geant4", "CC3": "CC3"}

PI0_MASS = 0.1349766
FINDER_WINDOW = (PI0_MASS - 0.04, PI0_MASS + 0.04)
BINS = np.linspace(0.090, 0.180, 46)          # 2 MeV bins
BINS_WIDE = np.linspace(0.070, 0.200, 66)     # 2 MeV bins, for the formula check
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])

WEAK, PHOTON, OTHER = 0, 1, 2
CLASSES = ["no_photon", "one_photon", "two_same", "two_distinct"]
SIM_SPLITS = ["both_g4", "one_each", "both_fastsim"]
PARENT_SPLITS = ["same_pi0", "different_pi0s", "one_from_pi0", "none_from_pi0"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 14,
    "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

LOG = []
def log(s=""):
    print(s, flush=True)
    LOG.append(s)

def tag(f):
    return "floor" + str(f).replace(".", "p")

# ------------------------------------------------------------------ loading
NEED = []
for k in (0, 1):
    NEED += [f"d{k}_{c}" for c in ("E", "px", "py", "pz", "mass_field")]
    NEED += [f"d{k}_l0_{c}" for c in ("cw", "pdg", "idx", "fast", "pi0_par_idx", "par0_pdg")]

def load(path):
    z = np.load(path, allow_pickle=False)
    d = {c: z[c] for c in NEED}
    d["meta_infile"] = str(z["meta_infile"])
    d["events_read"] = int(z["meta_events_read"])
    d["events_with_cand"] = int(z["meta_events_with_cand"])
    for c in ("ev_n_cands", "ev_n_gamma_pfo", "ev_n_gamma_pfo_p05"):
        if c in z.files:
            d[c] = z[c]
    d["n"] = len(d["d0_E"])
    return d

# ------------------------------------------------------------------ classification
def daughter_state(d, k, floor):
    cw, pdg = d[f"d{k}_l0_cw"], d[f"d{k}_l0_pdg"]
    state = np.full(len(cw), OTHER, dtype=np.int8)
    state[pdg == 22] = PHOTON
    state[cw < floor] = WEAK          # weak overrides the PDG; cw = -1 (no link) is weak too
    return state

def classify_pairs(d, floor):
    s0, s1 = daughter_state(d, 0, floor), daughter_state(d, 1, floor)
    n_photon = (s0 == PHOTON).astype(int) + (s1 == PHOTON).astype(int)
    same_mc = d["d0_l0_idx"] == d["d1_l0_idx"]
    cls = np.full(d["n"], -1, dtype=np.int8)
    cls[n_photon == 0] = 0
    cls[n_photon == 1] = 1
    cls[(n_photon == 2) & same_mc] = 2
    cls[(n_photon == 2) & ~same_mc] = 3
    assert (cls >= 0).all(), "some candidates got no class"
    n_states = sum(int((s == st).sum()) for s in (s0, s1) for st in (WEAK, PHOTON, OTHER))
    assert n_states == 2 * d["n"], "daughter states do not add up to 2 x candidates"
    assert sum(int((cls == i).sum()) for i in range(4)) == d["n"], "classes do not add up"
    return cls, s0, s1

def sim_split_masks(d):
    f0, f1 = d["d0_l0_fast"] == 1, d["d1_l0_fast"] == 1
    return {"both_g4": (~f0) & (~f1), "one_each": f0 ^ f1, "both_fastsim": f0 & f1}

def parent_split_masks(d):
    p0, p1 = d["d0_l0_pi0_par_idx"], d["d1_l0_pi0_par_idx"]
    h0, h1 = p0 >= 0, p1 >= 0
    return {"same_pi0": h0 & h1 & (p0 == p1), "different_pi0s": h0 & h1 & (p0 != p1),
            "one_from_pi0": h0 ^ h1, "none_from_pi0": (~h0) & (~h1)}

# ------------------------------------------------------------------ masses
def masses(d):
    E = d["d0_E"] + d["d1_E"]
    px = d["d0_px"] + d["d1_px"]; py = d["d0_py"] + d["d1_py"]; pz = d["d0_pz"] + d["d1_pz"]
    m2_full = E**2 - px**2 - py**2 - pz**2
    dot = d["d0_px"]*d["d1_px"] + d["d0_py"]*d["d1_py"] + d["d0_pz"]*d["d1_pz"]
    m2_massless = 2.0 * (d["d0_E"]*d["d1_E"] - dot)
    return (np.sqrt(np.clip(m2_full, 0, None)), np.sqrt(np.clip(m2_massless, 0, None)),
            int((m2_full < 0).sum()), int((m2_massless < 0).sum()))

# ------------------------------------------------------------------ plotting helpers
def draw(ax, m, label, color, ls="-"):
    counts, _ = np.histogram(m, bins=BINS)
    ax.stairs(counts, BINS, color=color, linewidth=2.0, linestyle=ls, label=label)
    ax.errorbar(CENTERS, counts, yerr=np.sqrt(counts), fmt="none", ecolor=color, elinewidth=0.9)

def decorate(ax, title):
    ax.axvline(PI0_MASS, color="gray", lw=0.9, ls=":")
    for e in FINDER_WINDOW:
        ax.axvline(e, color="gray", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlim(BINS[0], BINS[-1])
    ax.set_xlabel(r"$M_{\gamma\gamma}$ [GeV]  (formula A: full $E^2-|p|^2$)")
    ax.set_ylabel("candidates / 2 MeV")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", fontsize=10)

def footer(fig, floor=None):
    txt = f"Inputs: {os.path.basename(NPZ['G4'])} (G4), {os.path.basename(NPZ['CC3'])} (CC3). "
    txt += "Raw counts, error bars sqrt(N). Gray dashed = finder mass window 0.095-0.175 GeV, dotted = 0.135 GeV."
    if floor is not None:
        txt += f"\nLink weight floor f = {floor} (top-link cluster weight)."
    fig.text(0.5, -0.01, txt, ha="center", va="top", fontsize=9)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {name}")

# ------------------------------------------------------------------ plots
def plot_all(D, M):
    fig, ax = plt.subplots(figsize=(9, 6))
    for lab in LABELS:
        draw(ax, M[lab]["A"], f"{NAME[lab]} (N={D[lab]['n']:,}, {D[lab]['events_read']:,} events)", COLOR[lab],
             "-" if lab == "G4" else "--")
    decorate(ax, "Where do the finder's pi0 candidates sit in mass?\nAll candidates, no MC information")
    footer(fig)
    save(fig, "01_mass_all_candidates.png")

def plot_formula_check(D, M):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, lab in zip(axes, LABELS):
        for key, ls, nm in (("A", "-", "A full"), ("B", "--", "B massless")):
            c, _ = np.histogram(M[lab][key], bins=BINS_WIDE)
            ax.stairs(c, BINS_WIDE, color=COLOR[lab], linestyle=ls, linewidth=2, label=f"{NAME[lab]}, formula {nm}")
        for e in FINDER_WINDOW:
            ax.axvline(e, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(r"$M_{\gamma\gamma}$ [GeV]"); ax.set_ylabel("candidates / 2 MeV")
        ax.set_title("Does the massless formula leak outside the finder window?", fontsize=12)
        ax.legend(frameon=False); ax.set_ylim(bottom=0)
    footer(fig)
    save(fig, "02_formula_check_A_vs_B.png")

def plot_classes(D, M, C, floor):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, ci, cname in zip(axes.ravel(), range(4), CLASSES):
        for lab in LABELS:
            m = M[lab]["A"][C[lab][floor][0] == ci]
            draw(ax, m, f"{NAME[lab]} (N={len(m):,}, {100*len(m)/D[lab]['n']:.1f}% of all candidates)",
                 COLOR[lab], "-" if lab == "G4" else "--")
        decorate(ax, f"Class '{cname}'  (floor {floor})")
    fig.suptitle(f"How does the pair mass depend on what the two reco photons are linked to?  (floor {floor})",
                 fontsize=14, y=1.0)
    plt.tight_layout(); footer(fig, floor)
    save(fig, f"03_mass_by_class_{tag(floor)}.png")

def plot_sim_split(D, M, C, floor):
    lab = "CC3"
    cls = C[lab][floor][0]
    td = cls == 3
    masks = sim_split_masks(D[lab])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, sname in zip(axes, SIM_SPLITS):
        m = M[lab]["A"][td & masks[sname]]
        draw(ax, m, f"CC3 (N={len(m):,}, {100*len(m)/max(td.sum(),1):.1f}% of two_distinct)", COLOR[lab])
        decorate(ax, f"two_distinct, {sname}  (floor {floor})")
    fig.suptitle("Does the simulation mode of the two true photons change the mass?  CC3 file only (G4 file has no flag)",
                 fontsize=13, y=1.02)
    plt.tight_layout(); footer(fig, floor)
    save(fig, f"04_two_distinct_by_simulation_{tag(floor)}.png")

def plot_parentage(D, M, C, floor):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, pname in zip(axes.ravel(), PARENT_SPLITS):
        for lab in LABELS:
            td = C[lab][floor][0] == 3
            m = M[lab]["A"][td & parent_split_masks(D[lab])[pname]]
            draw(ax, m, f"{NAME[lab]} (N={len(m):,}, {100*len(m)/max(td.sum(),1):.1f}% of two_distinct)",
                 COLOR[lab], "-" if lab == "G4" else "--")
        decorate(ax, f"two_distinct, {pname}  (floor {floor})")
    fig.suptitle(f"Are the two linked true photons daughters of the same pi0?  (floor {floor})", fontsize=14, y=1.0)
    plt.tight_layout(); footer(fig, floor)
    save(fig, f"05_two_distinct_by_parentage_{tag(floor)}.png")

# ------------------------------------------------------------------ printed tables
def pct(a, b):
    return f"{100.0 * a / max(b, 1):5.1f}%"

def summarize(lab, d, mA, cls, s0, s1, floor):
    N = d["n"]
    log(f"\n--- {lab}  floor {floor}  (N candidates = {N:,}) ---")
    log("  daughter states (unit: daughters, denominator 2N = %d):" % (2 * N))
    for nm, st in (("photon", PHOTON), ("other", OTHER), ("weak", WEAK)):
        n = int((s0 == st).sum() + (s1 == st).sum())
        log(f"    {nm:7s} {n:8d}  {pct(n, 2*N)}")
    log("  pair classes (unit: candidates, denominator N):")
    for i, nm in enumerate(CLASSES):
        n = int((cls == i).sum())
        log(f"    {nm:13s} {n:8d}  {pct(n, N)}")
    for i in (0, 1):
        sel = cls == i
        non_photon_states = np.concatenate([s0[sel][s0[sel] != PHOTON], s1[sel][s1[sel] != PHOTON]])
        no, nw = int((non_photon_states == OTHER).sum()), int((non_photon_states == WEAK).sum())
        log(f"  {CLASSES[i]}: non-photon daughters (unit: daughters) = {len(non_photon_states)}: other {no}, weak {nw}")
    td = cls == 3
    ntd = int(td.sum())
    log(f"  inside two_distinct (N={ntd}):")
    sm = sim_split_masks(d)
    log("    simulation split: " + ", ".join(f"{k} {int((td & v).sum())} ({pct((td & v).sum(), ntd).strip()})" for k, v in sm.items()))
    pm = parent_split_masks(d)
    log("    parentage split:  " + ", ".join(f"{k} {int((td & v).sum())} ({pct((td & v).sum(), ntd).strip()})" for k, v in pm.items()))
    assert sum(int((td & v).sum()) for v in sm.values()) == ntd, "simulation split does not add up"
    assert sum(int((td & v).sum()) for v in pm.values()) == ntd, "parentage split does not add up"
    mism = 0
    for k in (0, 1):
        any_par = d[f"d{k}_l0_pi0_par_idx"] >= 0
        first_par = d[f"d{k}_l0_par0_pdg"] == 111
        mism += int(((any_par != first_par) & td).sum())
    log(f"    any-parent vs first-parent pi0 test disagree for {mism} of {2*ntd} daughters in two_distinct")
    log("  mass formula A, per class (mean/median/std of the TRUNCATED distribution, not a resolution):")
    for i, nm in enumerate(CLASSES):
        m = mA[cls == i]
        if len(m):
            log(f"    {nm:13s} N={len(m):7d}  mean {1000*m.mean():7.2f}  median {1000*np.median(m):7.2f}  std {1000*m.std():6.2f} MeV")

def compare_table(D, C, floor):
    log(f"\n=== Side by side, floor {floor}: class counts (% of all candidates in that file) ===")
    log(f"  {'class':13s} {'G4':>20s} {'CC3':>20s}")
    for i, nm in enumerate(CLASSES):
        row = []
        for lab in LABELS:
            n = int((C[lab][floor][0] == i).sum())
            row.append(f"{n:9d} ({pct(n, D[lab]['n']).strip():>6s})")
        log(f"  {nm:13s} {row[0]:>20s} {row[1]:>20s}")
    log(f"  {'all':13s} {D['G4']['n']:>20d} {D['CC3']['n']:>20d}")

def event_level(lab, d):
    log(f"\n--- {lab}: event level ---")
    log(f"  events read {d['events_read']:,}, with >=1 candidate {d['events_with_cand']:,} ({pct(d['events_with_cand'], d['events_read']).strip()})")
    if "ev_n_cands" not in d:
        log("  per-event columns not in file, skipped"); return
    nc = d["ev_n_cands"]
    log(f"  candidates per event (events with >=1): mean {nc[nc > 0].mean():.2f}, max {int(nc.max())}")
    z = nc == 0
    p05 = d["ev_n_gamma_pfo_p05"][z]
    log(f"  events with 0 candidates: {int(z.sum()):,}. Number of photon PFOs with p>=0.5 GeV in them:")
    for lo, hi, nm in ((0, 0, "0"), (1, 1, "1"), (2, 10**9, "2 or more")):
        n = int(((p05 >= lo) & (p05 <= hi)).sum())
        log(f"    {nm:10s} {n:7d}  {pct(n, int(z.sum())).strip()}")

# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    log(f"Output folder: {OUT}")
    D = {lab: load(NPZ[lab]) for lab in LABELS}
    for lab in LABELS:
        log(f"{lab}: {D[lab]['meta_infile']}  ({D[lab]['n']:,} candidates)")
    M, C = {}, {}
    for lab in LABELS:
        a, b, na, nb = masses(D[lab])
        M[lab] = {"A": a, "B": b}
        lo, hi = FINDER_WINDOW
        for key, arr, nneg in (("A", a, na), ("B", b, nb)):
            inside = float(((arr >= lo) & (arr <= hi)).mean())
            log(f"{lab} formula {key}: inside finder window {100*inside:.2f}%, M^2<0 for {nneg} candidates")
        C[lab] = {f: classify_pairs(D[lab], f) for f in FLOORS}

    log("\n=== Printed tables ===")
    for lab in LABELS:
        event_level(lab, D[lab])
    for f in FLOORS:
        for lab in LABELS:
            cls, s0, s1 = C[lab][f]
            summarize(lab, D[lab], M[lab]["A"], cls, s0, s1, f)
        compare_table(D, C, f)
    log(f"\nG4 file flagged fast-sim (top-link photons, all candidates, expected 0): "
        f"{int((D['G4']['d0_l0_fast'] == 1).sum() + (D['G4']['d1_l0_fast'] == 1).sum())}")

    log("\n=== Plots ===")
    plot_all(D, M)
    plot_formula_check(D, M)
    for f in FLOORS:
        plot_classes(D, M, C, f)
        plot_sim_split(D, M, C, f)
        plot_parentage(D, M, C, f)
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    log(f"\nSaved plots and summary.txt to {OUT}")

if __name__ == "__main__":
    main()
