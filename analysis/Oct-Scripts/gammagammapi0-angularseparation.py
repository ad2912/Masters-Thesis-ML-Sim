#!/usr/bin/env python3
"""
plots_gammagammapi0_v2.py
Plots and tables from the two GammaGammaCandidatePi0s .npz files (Geant4 vs CC3). Link weight floor 0.5.

Vocabulary (slide wording)
  candidate                         = one pair of reco photons from the finder
  two different true photons        = each reco photon is matched (cluster weight >= 0.5) to a true photon, and the two
                                      true photons are different particles
  both photons from the same true pi0 = as above, and both true photons have the same pi0 as parent
  pair energy                       = energy of the two reco photons added together (the reconstructed pi0 energy)
  invariant mass M(gamma gamma)     = sqrt( (E1+E2)^2 - |p1+p2|^2 ),  E in GeV, c = 1

Outputs (in PLOT_OUT)
  01  CC3 / G4 ratio per mass bin: raw yield and shape only, for two classes
  02  CC3 / G4 yield vs pair energy: candidates and unique true pi0s
  03  Mass shapes in pair-energy groups, G4 vs CC3 (unit area)
  04  Opening angle of the two reco photons, G4 vs CC3 and ratio, three energy groups
  05  Like-for-like mass shape: G4 (both photons reach the ECAL) vs CC3 both-fast-sim, same true-photon cuts
  06  Energy response and opening-angle response vs pair energy (all same-pi0 pairs)
  07  The same response for the like-for-like sample
  summary_v2.txt  all printed tables (yield by energy, event-by-event comparison, like-for-like counts)
Every step runs on its own: if one fails, the others still produce output.
Run: python3 plots_gammagammapi0_v2.py    (env overrides: G4_NPZ, CC3_NPZ, PLOT_OUT)
"""
import os
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ = {
    "G4":  os.environ.get("G4_NPZ",  "/data/dust/user/alimuham/thesis/reco/gammagammapi0-candidates-g4.npz"),
    "CC3": os.environ.get("CC3_NPZ", "/data/dust/user/alimuham/thesis/reco/gammagammapi0-candidates-cc3.npz"),
}
OUT = os.environ.get("PLOT_OUT", "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/gammagammapi0-angular-separation-100k")
FLOOR = 0.5
EDGES = [5, 10, 20, 30, 50, 125]                 # pair-energy groups [GeV]
LABELS = ["G4", "CC3"]
COLOR = {"G4": "#2166ac", "CC3": "#d6604d"}
NAME = {"G4": "Geant4", "CC3": "CC3"}
M0 = 0.1349766
WIN = (M0 - 0.04, M0 + 0.04)                     # finder mass window
BINS = WIN[0] + 0.002 * np.arange(41)            # 40 bins of 2 MeV, edges = window edges
CEN = 0.5 * (BINS[:-1] + BINS[1:])
BINS4 = WIN[0] + 0.004 * np.arange(21)           # 20 bins of 4 MeV
CEN4 = 0.5 * (BINS4[:-1] + BINS4[1:])
ANGLE_BINS = np.arange(0.0, 41.0, 1.0)           # mrad
ANGLE_GROUPS = [(10, 20), (30, 50), (50, 125)]
WEAK, PHOTON, OTHER = 0, 1, 2
PHI_CENTERS = [-157.5 + 45.0 * k for k in range(8)]
PA, PB = -0.01 - 0.5, 8.0 + 0.5
TXT = {"two": "two different true photons", "same": "both photons from the same true pi0"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 13,
    "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
LOG = []
def log(s=""):
    print(s, flush=True); LOG.append(s)

# ------------------------------------------------------------------ loading
NEED = ["evt"]
for k in (0, 1):
    NEED += [f"d{k}_{c}" for c in ("E", "px", "py", "pz")]
    NEED += [f"d{k}_l0_{c}" for c in ("cw", "pdg", "idx", "fast", "pi0_par_idx", "E", "theta", "phi", "gen", "any_par_e", "r_end")]

def load(path):
    z = np.load(path, allow_pickle=False)
    d = {c: z[c] for c in NEED}
    d["ev_evt"] = z["ev_evt"]
    d["n"] = len(d["d0_E"])
    return d

# ------------------------------------------------------------------ definitions
def state(d, k, floor):
    cw, pdg = d[f"d{k}_l0_cw"], d[f"d{k}_l0_pdg"]
    s = np.full(len(cw), OTHER, dtype=np.int8)
    s[pdg == 22] = PHOTON
    s[cw < floor] = WEAK
    return s

def two_distinct(d):
    s0, s1 = state(d, 0, FLOOR), state(d, 1, FLOOR)
    return (s0 == PHOTON) & (s1 == PHOTON) & (d["d0_l0_idx"] != d["d1_l0_idx"])

def same_pi0_only(d):
    p0, p1 = d["d0_l0_pi0_par_idx"], d["d1_l0_pi0_par_idx"]
    return (p0 >= 0) & (p1 >= 0) & (p0 == p1)

def both_fastsim(d):
    return (d["d0_l0_fast"] == 1) & (d["d1_l0_fast"] == 1)

def mass(d):
    E = d["d0_E"] + d["d1_E"]
    px = d["d0_px"] + d["d1_px"]; py = d["d0_py"] + d["d1_py"]; pz = d["d0_pz"] + d["d1_pz"]
    return np.sqrt(np.clip(E**2 - px**2 - py**2 - pz**2, 0, None))

def pair_energy(d):
    return d["d0_E"] + d["d1_E"]

def opening_angle_mrad(d):
    p0 = np.sqrt(d["d0_px"]**2 + d["d0_py"]**2 + d["d0_pz"]**2)
    p1 = np.sqrt(d["d1_px"]**2 + d["d1_py"]**2 + d["d1_pz"]**2)
    dot = d["d0_px"]*d["d1_px"] + d["d0_py"]*d["d1_py"] + d["d0_pz"]*d["d1_pz"]
    return 1000.0 * np.arccos(np.clip(dot / np.where(p0*p1 > 0, p0*p1, 1), -1, 1))

def true_angle_mrad(d):
    def unit(k):
        th = np.radians(d[f"d{k}_l0_theta"]); ph = np.radians(d[f"d{k}_l0_phi"])
        return np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
    a, b = unit(0), unit(1)
    return 1000.0 * np.arccos(np.clip(a[0]*b[0] + a[1]*b[1] + a[2]*b[2], -1, 1))

def energy_group(E, edges=EDGES):
    g = np.digitize(E, edges) - 1
    g[(E < edges[0]) | (E >= edges[-1])] = -1
    return g

def in_phi_crack(phi):
    dead = np.zeros(len(phi), dtype=bool)
    for c in PHI_CENTERS:
        dead |= ((phi - c - PA) % 360.0) <= (PB - PA)
    return dead

def true_photon_ok(d, k):
    p = f"d{k}_l0_"
    return ((d[p+"gen"] == 1) & (d[p+"E"] >= 10) & (d[p+"any_par_e"] == 0) &
            (d[p+"theta"] >= 40) & (d[p+"theta"] <= 140) & (~in_phi_crack(d[p+"phi"])))

def gname(i):
    return f"{EDGES[i]}-{EDGES[i+1]} GeV"

def ratio_err(a, b):
    """b/a with Poisson errors (arrays); nan where a or b is 0"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = (a > 0) & (b > 0)
    sa = np.where(a > 0, a, 1); sb = np.where(b > 0, b, 1)
    r = np.where(ok, b / sa, np.nan)
    e = r * np.sqrt(np.where(ok, 1 / sa + 1 / sb, np.nan))
    return r, e

# ------------------------------------------------------------------ plotting helpers
def draw(ax, m, label, color, ls="-", bins=BINS, norm=False):
    c, _ = np.histogram(m, bins=bins)
    n = max(len(m), 1) if norm else 1.0
    cen = 0.5 * (bins[:-1] + bins[1:])
    ax.stairs(c / n, bins, color=color, linewidth=2.0, linestyle=ls, label=label)
    ax.errorbar(cen, c / n, yerr=np.sqrt(c) / n, fmt="none", ecolor=color, elinewidth=0.9)

def finder_lines(ax):
    ax.axvline(M0, color="gray", lw=0.9, ls=":")
    for e in WIN:
        ax.axvline(e, color="gray", lw=0.7, ls="--", alpha=0.6)

def footer(fig, extra=""):
    txt = (f"Inputs: {os.path.basename(NPZ['G4'])} (G4), {os.path.basename(NPZ['CC3'])} (CC3). "
           f"Each reco photon matched to a true photon by cluster weight >= {FLOOR}. Error bars sqrt(N). "
           "Gray dashed = finder mass window, dotted = 0.135 GeV.")
    fig.text(0.5, -0.01, txt + ("\n" + extra if extra else ""), ha="center", va="top", fontsize=9)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig); log(f"  saved {name}")

def xpos(i, off=1.0):
    return np.sqrt(EDGES[i] * EDGES[i + 1]) * off

def energy_axis(ax):
    ax.set_xscale("log"); ax.set_xticks(EDGES); ax.set_xticklabels([str(e) for e in EDGES])
    ax.set_xlabel("pair energy (two reco photons added) [GeV]")

def mass_axis(ax, title):
    finder_lines(ax)
    ax.set_xlim(BINS[0], BINS[-1]); ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$M(\gamma\gamma)$ [GeV]"); ax.set_title(title, fontsize=11)

# ------------------------------------------------------------------ step 1: ratio per mass bin
def step_ratio(D, M, TD, SP):
    rows = [("two", TD), ("same", SP)]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    log("\n[01] CC3 / G4 per mass bin (4 MeV bins)")
    for r, (key, S) in enumerate(rows):
        a, _ = np.histogram(M["G4"][S["G4"]], bins=BINS4)
        b, _ = np.histogram(M["CC3"][S["CC3"]], bins=BINS4)
        Na, Nb = a.sum(), b.sum()
        rr, ee = ratio_err(a, b)
        for c, (y, yl, ttl, ref, sc) in enumerate((
                (rr, "CC3 / G4 (raw counts)", "raw yield", Nb / Na, 1.0),
                (rr * Na / Nb, "CC3 / G4 (each scaled to unit area)", "shape only", 1.0, Na / Nb))):
            ax = axes[r][c]
            ax.errorbar(CEN4, y, yerr=ee * sc, fmt="o", ms=4, color="black", ecolor="gray")
            ax.axhline(1.0, color="gray", lw=1.0)
            if c == 0:
                ax.axhline(ref, color=COLOR["CC3"], lw=1.2, ls="--", label=f"overall ratio {ref:.3f}")
                ax.legend(frameon=False, loc="upper left")
            finder_lines(ax)
            ax.set_xlim(BINS4[0], BINS4[-1]); ax.set_ylim(0.5, 1.5)
            ax.set_xlabel(r"$M(\gamma\gamma)$ [GeV]"); ax.set_ylabel(yl)
            ax.set_title(f"{TXT[key]}: {ttl}\n(G4 N={Na:,}, CC3 N={Nb:,})", fontsize=11)
        log(f"  {TXT[key]}: G4 N={Na}, CC3 N={Nb}, overall ratio {Nb/Na:.4f}")
    fig.suptitle("Where does CC3 differ from Geant4 in the pi0 mass distribution?", fontsize=14, y=1.0)
    plt.tight_layout(); footer(fig, "Ratio errors from Poisson counts; G4 and CC3 share generator events, so they are approximate.")
    save(fig, "01_ratio_cc3_over_g4.png")

# ------------------------------------------------------------------ step 2: yield vs pair energy
def unique_pi0_keys(d, sel):
    ev = d["evt"][sel].astype(np.int64); pi = d["d0_l0_pi0_par_idx"][sel].astype(np.int64)
    ukey, first = np.unique(ev * 1_000_000 + pi, return_index=True)
    return ukey, np.flatnonzero(sel)[first]

def step_yield(D, SP):
    ng = len(EDGES) - 1
    cand, uniq, U = {}, {}, {}
    for lab in LABELS:
        g = energy_group(pair_energy(D[lab]))
        cand[lab] = np.array([int((SP[lab] & (g == i)).sum()) for i in range(ng)])
        ukey, idx = unique_pi0_keys(D[lab], SP[lab])
        gu = energy_group(pair_energy(D[lab])[idx])
        uniq[lab] = np.array([int((gu == i).sum()) for i in range(ng)])
        U[lab] = ukey
        log(f"  {lab}: both-photons-from-same-pi0 candidates {int(SP[lab].sum()):,} -> unique true pi0s {len(ukey):,} "
            f"({SP[lab].sum()/max(len(ukey),1):.2f} candidates per pi0)")
    log("\n[02] CC3 / G4 yield by pair energy (both photons from the same true pi0)")
    log(f"  {'group':>10s} {'cand G4':>8s} {'cand CC3':>9s} {'ratio':>14s} | {'pi0 G4':>7s} {'pi0 CC3':>8s} {'ratio':>14s}")
    rc, ec = ratio_err(cand["G4"], cand["CC3"]); ru, eu = ratio_err(uniq["G4"], uniq["CC3"])
    for i in range(ng):
        log(f"  {gname(i):>10s} {cand['G4'][i]:8d} {cand['CC3'][i]:9d} {rc[i]:7.3f}+-{ec[i]:5.3f} | "
            f"{uniq['G4'][i]:7d} {uniq['CC3'][i]:8d} {ru[i]:7.3f}+-{eu[i]:5.3f}")
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for (r, e, off, col, mk, nm) in ((rc, ec, 0.96, "gray", "s", "candidates"),
                                      (ru, eu, 1.04, "black", "o", "unique true pi0s")):
        xs = [xpos(i, off) for i in range(ng)]
        xe = np.array([[xpos(i) - EDGES[i], EDGES[i + 1] - xpos(i)] for i in range(ng)]).T * 0.9
        ax.errorbar(xs, r, yerr=e, xerr=xe, fmt=mk, color=col, capsize=3, label=nm)
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_ylim(0.7, 1.15); energy_axis(ax)
    ax.set_ylabel("CC3 / Geant4 yield")
    ax.set_title("Does CC3 find fewer correctly paired pi0s as the pi0 energy grows?")
    ax.legend(frameon=False, loc="lower left")
    footer(fig, "Both photons from the same true pi0. Unique true pi0 = one count per (event, true pi0). Group from reco pair energy.")
    save(fig, "02_yield_ratio_vs_pair_energy.png")
    return U

# ------------------------------------------------------------------ step 3: shapes by energy
def step_shapes(D, M, SP):
    ng = len(EDGES) - 1
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for i, ax in enumerate(axes.ravel()):
        if i >= ng:
            ax.axis("off"); continue
        for lab in LABELS:
            g = energy_group(pair_energy(D[lab]))
            m = M[lab][SP[lab] & (g == i)]
            draw(ax, m, f"{NAME[lab]} (N={len(m):,})", COLOR[lab], "-" if lab == "G4" else "--", norm=True)
        mass_axis(ax, f"pair energy {gname(i)}")
        ax.set_ylabel("fraction of pairs in group / 2 MeV"); ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Pi0 mass shape in pair-energy groups: Geant4 vs CC3 (each curve scaled to unit area)\n"
                 "both photons from the same true pi0", fontsize=13, y=1.0)
    plt.tight_layout(); footer(fig)
    save(fig, "03_mass_shapes_by_pair_energy.png")

# ------------------------------------------------------------------ step 4: opening angle
def step_angle(D, SP):
    fig, axes = plt.subplots(2, len(ANGLE_GROUPS), figsize=(6 * len(ANGLE_GROUPS), 9), sharex=True)
    log("\n[04] opening angle of the two reco photons (both photons from the same true pi0)")
    cen = 0.5 * (ANGLE_BINS[:-1] + ANGLE_BINS[1:])
    for j, (lo, hi) in enumerate(ANGLE_GROUPS):
        H, ANG, SEL = {}, {}, {}
        for lab in LABELS:
            SEL[lab] = SP[lab] & (pair_energy(D[lab]) >= lo) & (pair_energy(D[lab]) < hi)
            ANG[lab] = opening_angle_mrad(D[lab])[SEL[lab]]
            H[lab], _ = np.histogram(ANG[lab], bins=ANGLE_BINS)
            axes[0][j].stairs(H[lab], ANGLE_BINS, color=COLOR[lab], linewidth=2, linestyle="-" if lab == "G4" else "--",
                              label=f"{NAME[lab]} (N={len(ANG[lab]):,})")
        r, e = ratio_err(H["G4"], H["CC3"])
        axes[1][j].errorbar(cen, r, yerr=e, fmt="o", ms=3, color="black", ecolor="gray")
        tot = H["CC3"].sum() / max(H["G4"].sum(), 1)
        axes[1][j].axhline(1, color="gray"); axes[1][j].axhline(tot, color=COLOR["CC3"], ls="--", label=f"overall {tot:.3f}")
        axes[1][j].set_ylim(0.4, 1.6); axes[1][j].legend(frameon=False)
        axes[1][j].set_xlabel("opening angle of the two reco photons [mrad]"); axes[1][j].set_ylabel("CC3 / G4")
        axes[0][j].set_ylabel("pairs / mrad"); axes[0][j].set_title(f"pair energy {lo}-{hi} GeV")
        axes[0][j].legend(frameon=False)
        med = np.median(ANG["G4"]) if len(ANG["G4"]) else np.nan
        sm = [(ANG[lab] < med).sum() for lab in LABELS]; lg = [(ANG[lab] >= med).sum() for lab in LABELS]
        r1, e1 = ratio_err([sm[0]], [sm[1]]); r2, e2 = ratio_err([lg[0]], [lg[1]])
        log(f"  {lo}-{hi} GeV: yield ratio overall {tot:.3f}; angle below G4 median ({med:.1f} mrad) {r1[0]:.3f}+-{e1[0]:.3f}; "
            f"above {r2[0]:.3f}+-{e2[0]:.3f}")
    fig.suptitle("Is the CC3 yield deficit concentrated at small opening angles (showers merging)?", fontsize=13, y=1.0)
    plt.tight_layout(); footer(fig, "Both photons from the same true pi0. Opening angle from the reco photon momenta.")
    save(fig, "04_opening_angle.png")

# ------------------------------------------------------------------ step 5: like-for-like
def reaches_ecal(d, k):
    """true photon did not end before the ECAL (end radius >= 1780 mm; a photon with no stored endpoint counts as reaching)"""
    return ~(d[f"d{k}_l0_r_end"] < 1780.0)

def step_like_for_like(D, M, TD, SP):
    OK = {lab: true_photon_ok(D[lab], 0) & true_photon_ok(D[lab], 1) for lab in LABELS}
    REACH = {lab: reaches_ecal(D[lab], 0) & reaches_ecal(D[lab], 1) for lab in LABELS}
    FAST = both_fastsim(D["CC3"])
    log("\n[05] like-for-like. True-photon cuts on both files: status 1, E >= 10 GeV, no e+- parent, barrel theta 40-140 deg,"
        " outside phi cracks.")
    log("     CC3 fast-sim only acts on photons that reach the ECAL, so the G4 reference also requires both true photons"
        " to reach it (end radius >= 1780 mm).")
    for nm, S in (("two different true photons", TD), ("both photons from the same true pi0", SP)):
        g_cuts = int((S["G4"] & OK["G4"]).sum()); g_reach = int((S["G4"] & OK["G4"] & REACH["G4"]).sum())
        c_cuts = int((S["CC3"] & OK["CC3"]).sum()); c_fast = int((S["CC3"] & OK["CC3"] & FAST).sum())
        r_all, e_all = ratio_err([g_cuts], [c_cuts]); r_ll, e_ll = ratio_err([g_reach], [c_fast])
        log(f"  {nm}:")
        log(f"    G4 passing the cuts {g_cuts:,}; of these both photons reach the ECAL {g_reach:,} ({100*g_reach/max(g_cuts,1):.1f}%)")
        log(f"    CC3 passing the cuts {c_cuts:,}; of these both photons fast-simulated {c_fast:,} ({100*c_fast/max(c_cuts,1):.1f}%)")
        log(f"    CC3/G4, passing the cuts only: {r_all[0]:.3f}+-{e_all[0]:.3f};  like-for-like (G4 reaching ECAL vs CC3 fast-sim): "
            f"{r_ll[0]:.3f}+-{e_ll[0]:.3f}")
    LFL = {"G4": SP["G4"] & OK["G4"] & REACH["G4"], "CC3": SP["CC3"] & OK["CC3"] & FAST}
    fig, ax = plt.subplots(figsize=(9, 6))
    for lab in LABELS:
        m = M[lab][LFL[lab]]
        nm = NAME[lab] + (" (both photons fast-simulated)" if lab == "CC3" else " (both photons reach the ECAL)")
        draw(ax, m, f"{nm}\nN={len(m):,}, median {1000*np.median(m):.1f} MeV" if len(m) else nm, COLOR[lab],
             "-" if lab == "G4" else "--", norm=True)
    mass_axis(ax, "Same true-photon cuts on both files\nboth photons from the same true pi0")
    ax.set_ylabel("fraction of pairs / 2 MeV"); ax.legend(frameon=False, loc="upper left", fontsize=9)
    footer(fig, "True-photon cuts: status 1, E >= 10 GeV, no e+- parent, barrel theta 40-140 deg, outside phi cracks.")
    save(fig, "05_like_for_like_mass_shape.png")
    return LFL

# ------------------------------------------------------------------ step 6/7: response
def robust(v):
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return (np.nan, np.nan, np.nan, np.nan, len(v))
    s68 = 0.5 * (np.percentile(v, 84) - np.percentile(v, 16))
    return (np.median(v), 1.2533 * s68 / np.sqrt(len(v)), s68, s68 / np.sqrt(2 * len(v)), len(v))

def step_response(D, SELS, title, fname, tag):
    ng = len(EDGES) - 1
    S = {lab: {"E": [], "A": []} for lab in LABELS}
    for lab in LABELS:
        d = D[lab]; g = energy_group(pair_energy(d))
        ang_r, ang_t = opening_angle_mrad(d), true_angle_mrad(d)
        ratioA = np.where(ang_t > 0.5, ang_r / np.where(ang_t > 0, ang_t, 1), np.nan)
        for i in range(ng):
            sel = SELS[lab] & (g == i)
            def er(k):
                t = d[f"d{k}_l0_E"][sel]
                return np.where(t > 0, d[f"d{k}_E"][sel] / np.where(t > 0, t, 1), np.nan)
            S[lab]["E"].append(robust(np.concatenate([er(0), er(1)])))
            S[lab]["A"].append(robust(ratioA[sel]))
    log(f"\n[{tag}] {title}: median and sigma68 (half of the 16-84% range), no fits")
    log(f"  {'group':>10s} {'':>4s} | {'E median':>15s} {'E sigma68':>15s} | {'angle median':>15s} {'angle sigma68':>15s} | N")
    for i in range(ng):
        for lab in LABELS:
            e, a = S[lab]["E"][i], S[lab]["A"][i]
            log(f"  {gname(i):>10s} {lab:>4s} | {e[0]:7.4f}+-{e[1]:6.4f} {e[2]:7.4f}+-{e[3]:6.4f} | "
                f"{a[0]:7.4f}+-{a[1]:6.4f} {a[2]:7.4f}+-{a[3]:6.4f} | {a[4]}")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    spec = [(axes[0][0], "E", 0, 1, "median E_reco / E_true (per photon)", 1.0),
            (axes[0][1], "E", 2, 3, "spread of E_reco / E_true (half of central 68%)", None),
            (axes[1][0], "A", 0, 1, "median reco angle / true angle", 1.0),
            (axes[1][1], "A", 2, 3, "spread of reco angle / true angle (half of central 68%)", None)]
    for ax, key, iv, ie, yl, ref in spec:
        for lab, off in (("G4", 0.97), ("CC3", 1.03)):
            xs = [xpos(i, off) for i in range(ng)]
            ys = [S[lab][key][i][iv] for i in range(ng)]; es = [S[lab][key][i][ie] for i in range(ng)]
            ax.errorbar(xs, ys, yerr=es, fmt="o", color=COLOR[lab], capsize=3, label=NAME[lab])
        if ref is not None:
            ax.axhline(ref, color="gray", ls=":")
        energy_axis(ax); ax.set_ylabel(yl); ax.legend(frameon=False)
    fig.suptitle(title, fontsize=13, y=1.0)
    plt.tight_layout(); footer(fig, "True values from the linked true photons. Angle response needs a true angle above 0.5 mrad.")
    save(fig, fname)

# ------------------------------------------------------------------ step 8: event by event
def step_events(D, SP, U):
    log("\n[08] event by event: unique true pi0s found per event (events present in both files)")
    cnt = {}
    for lab in LABELS:
        evs, c = np.unique(U[lab] // 1_000_000, return_counts=True)
        cnt[lab] = dict(zip(evs.tolist(), c.tolist()))
    common = np.intersect1d(D["G4"]["ev_evt"].astype(np.int64), D["CC3"]["ev_evt"].astype(np.int64))
    a = np.array([cnt["G4"].get(int(e), 0) for e in common]); b = np.array([cnt["CC3"].get(int(e), 0) for e in common])
    log(f"  common events: {len(common):,} (G4 {len(D['G4']['ev_evt']):,}, CC3 {len(D['CC3']['ev_evt']):,})")
    log(f"  total unique true pi0s in common events: G4 {a.sum():,}, CC3 {b.sum():,}")
    log(f"  events with MORE pi0s in G4: {int((a > b).sum()):,}  (CC3 loses {int((a - b)[a > b].sum()):,} pi0s in them)")
    log(f"  events with the same number: {int((a == b).sum()):,}")
    log(f"  events with MORE pi0s in CC3: {int((b > a).sum()):,}  (CC3 gains {int((b - a)[b > a].sum()):,} pi0s in them)")

# ------------------------------------------------------------------ main
def run_step(name, fn, *args):
    try:
        return fn(*args)
    except Exception:
        log(f"\n!!! step '{name}' failed, continuing with the others:")
        log(traceback.format_exc())
        return None

def main():
    os.makedirs(OUT, exist_ok=True)
    log(f"Output folder: {OUT}")
    D = {lab: load(NPZ[lab]) for lab in LABELS}
    M = {lab: mass(D[lab]) for lab in LABELS}
    TD = {lab: two_distinct(D[lab]) for lab in LABELS}
    SP = {lab: TD[lab] & same_pi0_only(D[lab]) for lab in LABELS}
    for lab in LABELS:
        inside = float(((M[lab] >= WIN[0]) & (M[lab] <= WIN[1])).mean())
        log(f"{lab}: {D[lab]['n']:,} candidates, {100*inside:.2f}% inside the finder window")
    run_step("ratio", step_ratio, D, M, TD, SP)
    U = run_step("yield", step_yield, D, SP)
    run_step("shapes", step_shapes, D, M, SP)
    LFL = run_step("like-for-like", step_like_for_like, D, M, TD, SP)
    run_step("angle", step_angle, D, SP)
    run_step("response all", step_response, D, SP, "Energy and opening-angle response, both photons from the same true pi0",
             "06_energy_and_angle_response.png", "06")
    if LFL is not None:
        run_step("response like-for-like", step_response, D, LFL,
                 "Energy and opening-angle response, like-for-like sample (CC3: both photons fast-simulated)",
                 "07_response_like_for_like.png", "07")
    if U is not None:
        run_step("events", step_events, D, SP, U)
    with open(os.path.join(OUT, "summary_v2.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    log(f"\nSaved to {OUT}")

if __name__ == "__main__":
    main()
