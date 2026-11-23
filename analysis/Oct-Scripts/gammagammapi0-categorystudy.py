#!/usr/bin/env python3
"""
category_study.py -- reco-side category study, Geant4 (G4) vs CC3, true pi0 -> two photons.

RUN (no ROOT needed, reads the NPZ tables written by make_pi0_tables.py):
  python3 category_study.py                          0.04 tables  pi0-tables-{g4,cc3}-full.npz
  python3 category_study.py --suffix full-dmgg0.1    0.1 tables (finder window +-0.1 GeV is taken from the suffix)
  python3 category_study.py --suffix first1000       quick test on the 1k tables
  options: --case {full,barrel,both}  --pairs resolved found  --min-pairs 100  --min-bin 10  --nbins 30
           --mass-bin 20 (MeV)  --floor 0.5  --maxdelta X (GeV)  --outdir DIR  --dpi 130  --g4 FILE --cc3 FILE

CATEGORIES (names as agreed; photons A and B of a generator pi0 P):
  CC3 file : count the photons with isHandledByFastSim()==1 :  2 = full fast-sim, 1 = partial fast-sim (EXACTLY one), 0 = fallback G4
  G4 file  : no flag exists, so count the photons with E_mc >= 10 GeV (and genStatus 1): 2 / 1 / 0 with the same three names.
  fallback G4 here = every pi0 of the file that CC3 did not handle (NOT the old isolated-trigger-space group of the 2026-10 photon study).
CASES:
  full   : all generator pi0s, no angular cuts.
  barrel : only pi0s whose BOTH photons are in the barrel acceptance (theta 40-140 deg, phi outside every one-sided dead window
           [C-0.51, C+8.5] deg, C = -157.5 + 45k deg). The categories are then counted as above.
POPULATION (as in plot_pi0_october29.py): true pi0 with two photon daughters, both genStatus 1 (generator pi0s).
PAIR STATES (floor 0.5 default): matched PFO = best PFO of a photon (rank cw*E_pfo) with cw >= floor.
  resolved = both photons matched to two DIFFERENT PFOs, both PDG 22.   res_E = resolved and both PFO E >= 0.5 GeV.
  res_E_window = res_E and unfitted pair mass inside 0.1349766 +- maxdelta.   found = the finder made a candidate from exactly these two PFOs.
PAIR MASS = invariant mass of the two matched PFOs: m = sqrt((E1+E2)^2 - |p1+p2|^2), the UNFITTED mass (not the finder's constrained fit).
ANGLES: theta_true = stored true_open_angle (the two generator photons); theta_reco = angle between the momenta of the two matched PFOs,
  atan2(|a x b|, a.b). r = theta_reco/theta_true. dtheta = theta_reco - theta_true in mrad.

OUTPUT (folder = PLOT_DIR/category-study-<suffix> unless --outdir): summary.txt (everything printed), counts_<case>.csv,
  stage_counts_<case>.md (CC3 / G4 cells, same columns as the plot B table), crosstab_<case>.csv, region_flag_fraction.csv,
  angle_summary_<case>.csv, found_fraction_vs_mass_<case>.csv, and the PNGs G_*, ratio_*, dtheta_*.

NUANCES TO KEEP IN MIND
  1. Energy axis = TRUE pi0 energy (needs the truth denominator). Bin edges 5,10,20,30,50,125 GeV, lower edge inclusive. Bins <5 and >=125 are in the CSVs only.
  2. 'full fast-sim' needs both photons >= 10 GeV, so it only exists for pi0 energy >= ~20 GeV; lower bins are empty by construction.
  3. In the FULL case the CC3 flag is 0 in the gaps and forward regions even for E>=10 GeV photons, while the G4 energy cut puts them in 'full'.
     So CC3 'fallback G4' (no flag) is NOT the same set as G4 'fallback G4' (no photon >= 10 GeV). The cross-tab and region table quantify this.
  4. CC3 'fallback G4' contains pi0s with photons below 10 GeV, outside the acceptance, AND early-converting photons; the G4 file has the same early conversions
     inside whichever energy class they fall in. No endpoint cut is applied (agreed to cut corners).
  5. Categories come from TRUE photons; Pandora reconstructing a fast-simulated photon below 10 GeV does not change the category.
  6. Flag value -1 (unreadable) is counted as 'not flagged' and reported.
  7. A panel is drawn only if BOTH files have >= --min-pairs pairs in it; histogram bins need >= --min-bin entries in both files for a ratio point.
  8. Histograms are normalised to unit area over the drawn x-range; entries beyond the range are counted in N and listed as 'beyond axis'.
     x-ranges are taken per energy bin from the pooled 0.5-99.5 percentiles of both files and all categories (same range in a column).
  9. Ratio panel: CC3/G4 of the normalised bin contents, independent-Poisson error on the two bin counts (approximate: shared generator events, normalisation covariance ignored).
 10. Found pairs are a subset of resolved pairs shaped by the finder window; use 'resolved' to study the response, 'found' for what the finder kept.
 11. Plot G: pairs outside the finder window have found fraction 0 by construction; dashed lines mark the window edges you gave with --maxdelta.
 12. --maxdelta must match the finder run of the tables (0.04 or 0.1). The self-check 'found pairs outside the window' must be 0 for the matching value.
 13. No systematics, no G4-vs-G4 baseline; the files share generator events, so ratio errors are approximate.
 14. Tested on synthetic tables only (checks the software, not the physics or the real files).
"""
import argparse
import csv
import math
import os
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REC_DIR = "/data/dust/user/alimuham/thesis/reco"
PLOT_DIR = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots"
E_EDGES = [5.0, 10.0, 20.0, 30.0, 50.0, 125.0]
E_NAMES = ["<5", "5-10", "10-20", "20-30", "30-50", "50-125", ">=125"]
DRAW_BINS = [1, 2, 3, 4, 5]
E_CUT = 0.5
M_PI0 = 0.1349766
EPS = 1e-6
E_PHOTON_MIN = 10.0
THETA_BARREL = (40.0, 140.0)
CRACK_CENTERS = [-157.5 + 45.0 * k for k in range(8)]
CRACK_LO, CRACK_HI = -0.51, 8.5
G4_COL, CC3_COL = "#2166ac", "#d6604d"
CATEGORIES = ["full fast-sim", "partial fast-sim", "fallback G4"]
CATEGORY_SOURCE = {"CC3": "the fast-sim flags of the two true photons",
                   "G4": "E_mc >= 10 GeV of the two true photons (no flag exists in the G4 file)"}
CASE_TEXT = {"full": "full angular space (all generator pi0s, no angular cuts)",
             "barrel": "barrel only (both photons at theta 40-140 deg and outside the phi dead windows, 0.5 deg margin)"}
GROUP_TEXT = {"resolved": "all pairs with two separate photon PFOs",
              "found": "pairs the finder kept (found)"}
GROUP_KEY = {"resolved": "res", "found": "found"}
STAGES = [("N_true", "all generator pi0s"), ("res", "two separate photon PFOs"),
          ("res_E", "+ both PFO energies >= 0.5 GeV"), ("res_E_window", "+ pair mass inside the finder window"),
          ("found", "found by the finder")]
LOG_EXPECT = {"full": {"G4": (176406, 60907), "CC3": (176402, 58038)},
              "full-dmgg0.1": {"G4": (176406, 61715), "CC3": (176402, 58946)}}

REQUIRED = ["mc_loop_evt", "mc_pi0_E", "mc_true_open_angle", "cand_loop_evt", "cand_d0_pfo_idx", "cand_d1_pfo_idx"]
for _k in (0, 1):
    for _c in ("gen", "E", "theta", "phi", "fast"):
        REQUIRED.append("mc_ph%d_%s" % (_k, _c))
    for _c in ("pfo_idx", "cw", "pdg", "E", "px", "py", "pz"):
        REQUIRED.append("mc_ph%d_b0_%s" % (_k, _c))


class Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "w"), sys.__stdout__

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def bin_index(x, edges=E_EDGES):
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        b = np.digitize(x, edges)
    b[~np.isfinite(x)] = -1
    return b


def pair_key(loop, a, b):
    loop, a, b = (np.asarray(v, dtype=np.int64) for v in (loop, a, b))
    if a.size and (max(a.max(), b.max()) >= 2 ** 21):
        raise ValueError("PFO index too large for the key packing")
    return (loop << 42) | (np.minimum(a, b) << 21) | np.maximum(a, b)


def binom_err(k, n):
    return math.sqrt(max(k / n * (1 - k / n), 0.0) / n) if n > 0 else float("nan")


def s68(v):
    p16, p84 = np.percentile(v, [16, 84])
    return 0.5 * (p84 - p16)


# ----------------------------------------------------------------------------------------------
# photon-level geometry
# ----------------------------------------------------------------------------------------------
def phi_in_dead_window(phi_deg):
    """True where phi lies in the one-sided dead window [C-0.51, C+8.5] deg of any crack centre C."""
    phi = np.asarray(phi_deg, dtype=float)
    dead = np.zeros(phi.shape, dtype=bool)
    for c in CRACK_CENTERS:
        raw = ((phi - c + 180.0) % 360.0) - 180.0
        dead |= (raw >= CRACK_LO) & (raw <= CRACK_HI)
    return dead


def photon_in_barrel(theta_deg, phi_deg):
    th = np.asarray(theta_deg, dtype=float)
    ph = np.asarray(phi_deg, dtype=float)
    ok = np.isfinite(th) & np.isfinite(ph)
    return ok & (th >= THETA_BARREL[0]) & (th <= THETA_BARREL[1]) & ~phi_in_dead_window(ph)


def category_index(n_photons):
    """2 photons -> 0 (full fast-sim), 1 -> 1 (partial fast-sim), 0 -> 2 (fallback G4)."""
    n = np.asarray(n_photons)
    return np.where(n == 2, 0, np.where(n == 1, 1, 2))


# ----------------------------------------------------------------------------------------------
# loading and per-file state
# ----------------------------------------------------------------------------------------------
def load_tables(path):
    z = np.load(path)
    a = {k: z[k] for k in z.files if k.startswith(("mc_", "cand_"))}
    missing = [c for c in REQUIRED if c not in a]
    if missing:
        raise KeyError("columns missing in %s: %s" % (path, missing))
    return a


def candidate_keys(a):
    ok = (a["cand_d0_pfo_idx"] >= 0) & (a["cand_d1_pfo_idx"] >= 0)
    return pair_key(a["cand_loop_evt"][ok], a["cand_d0_pfo_idx"][ok], a["cand_d1_pfo_idx"][ok])


def build_state(a, tag, floor, maxdelta):
    n = len(a["mc_loop_evt"])
    ia = a["mc_ph0_b0_pfo_idx"].astype(np.int64)
    ib = a["mc_ph1_b0_pfo_idx"].astype(np.int64)
    ma = (ia >= 0) & (a["mc_ph0_b0_cw"].astype(float) >= floor - EPS)
    mb = (ib >= 0) & (a["mc_ph1_b0_cw"].astype(float) >= floor - EPS)
    both = ma & mb
    merged = both & (ia == ib)
    other = both & ~merged & ((a["mc_ph0_b0_pdg"] != 22) | (a["mc_ph1_b0_pdg"] != 22))
    res = both & ~merged & ~other
    v0 = np.stack([a["mc_ph0_b0_px"], a["mc_ph0_b0_py"], a["mc_ph0_b0_pz"]], axis=1).astype(float)
    v1 = np.stack([a["mc_ph1_b0_px"], a["mc_ph1_b0_py"], a["mc_ph1_b0_pz"]], axis=1).astype(float)
    e0, e1 = a["mc_ph0_b0_E"].astype(float), a["mc_ph1_b0_E"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        vs = v0 + v1
        m2 = (e0 + e1) ** 2 - np.sum(vs ** 2, axis=1)
        mass = np.sqrt(np.clip(m2, 0.0, None))
        theta_reco = np.arctan2(np.linalg.norm(np.cross(v0, v1), axis=1), np.sum(v0 * v1, axis=1)) * 1000.0
        theta_true = a["mc_true_open_angle"].astype(float) * 1000.0
        r = np.where(theta_true > 0, theta_reco / theta_true, np.nan)
        dtheta = theta_reco - theta_true
        neg_m2 = int((res & (m2 < 0)).sum())
        res_E = res & (e0 >= E_CUT) & (e1 >= E_CUT)
        res_E_window = res_E & (mass >= M_PI0 - maxdelta) & (mass <= M_PI0 + maxdelta)
    found = np.zeros(n, dtype=bool)
    idx = np.where(res)[0]
    if idx.size:
        found[idx] = np.isin(pair_key(a["mc_loop_evt"][idx], ia[idx], ib[idx]), candidate_keys(a))
    gen = (a["mc_ph0_gen"] == 1) & (a["mc_ph1_gen"] == 1)
    in_barrel = photon_in_barrel(a["mc_ph0_theta"], a["mc_ph0_phi"]) & photon_in_barrel(a["mc_ph1_theta"], a["mc_ph1_phi"])
    n_flag = (a["mc_ph0_fast"] == 1).astype(int) + (a["mc_ph1_fast"] == 1).astype(int)
    n_cut = (((a["mc_ph0_gen"] == 1) & (a["mc_ph0_E"] >= E_PHOTON_MIN)).astype(int)
             + ((a["mc_ph1_gen"] == 1) & (a["mc_ph1_E"] >= E_PHOTON_MIN)).astype(int))
    cat_flag, cat_cut = category_index(n_flag), category_index(n_cut)
    return dict(tag=tag, a=a, n=n, res=res, res_E=res_E, res_E_window=res_E_window, found=found,
                mass=mass, theta_reco=theta_reco, theta_true=theta_true, r=r, dtheta=dtheta, neg_m2=neg_m2,
                case={"full": gen, "barrel": gen & in_barrel}, gen=gen, in_barrel=in_barrel,
                n_flag=n_flag, cat_flag=cat_flag, cat_cut=cat_cut,
                cat=(cat_flag if tag == "CC3" else cat_cut),
                n_unknown_flag=int(((a["mc_ph0_fast"] < 0) | (a["mc_ph1_fast"] < 0))[gen].sum()),
                bin_true=bin_index(a["mc_pi0_E"]))


def cell_mask(S, case, ci=None, b=None):
    m = S["case"][case]
    if ci is not None:
        m = m & (S["cat"] == ci)
    if b is not None:
        m = m & (np.isin(S["bin_true"], b) if isinstance(b, (list, tuple)) else (S["bin_true"] == b))
    return m


def stage_counts(S, case, ci, b):
    m = cell_mask(S, case, ci, b)
    return [int(m.sum())] + [int((m & S[k]).sum()) for k in ("res", "res_E", "res_E_window", "found")]


def energy_rows():
    return [(E_NAMES[b], b) for b in DRAW_BINS] + [("5-125 (drawn bins)", list(DRAW_BINS)), ("all energies", None)]


# ----------------------------------------------------------------------------------------------
# 1. counts
# ----------------------------------------------------------------------------------------------
def fmt(n):
    return "{:,}".format(int(n))


def counts_outputs(Sc, Sg, case, outdir):
    rows, md = [], []
    for ci, cname in [(None, "all generator pi0s in this case")] + list(enumerate(CATEGORIES)):
        md.append("\n### %s | %s\n" % (CASE_TEXT[case], cname))
        md.append("| true E [GeV] | " + " | ".join(s[1] for s in STAGES) + " |")
        md.append("|---|" + "---|" * len(STAGES))
        for ename, b in energy_rows():
            vc, vg = stage_counts(Sc, case, ci, b), stage_counts(Sg, case, ci, b)
            md.append("| %s | " % ename + " | ".join("%s / %s" % (fmt(x), fmt(y)) for x, y in zip(vc, vg)) + " |")
            rows.append([case, cname, ename] + vc + vg +
                        [(vc[4] / vc[0] if vc[0] else float("nan")), (vg[4] / vg[0] if vg[0] else float("nan"))])
    hdr = ["case", "category", "true_E_bin"] + ["CC3_" + s[0] for s in STAGES] + ["G4_" + s[0] for s in STAGES] + ["eff_found_CC3", "eff_found_G4"]
    write_csv(os.path.join(outdir, "counts_%s.csv" % case), hdr, rows)
    text = "Cells are CC3 / G4. Each stage is cumulative. Categories: CC3 from %s; G4 from %s.\n" % (CATEGORY_SOURCE["CC3"], CATEGORY_SOURCE["G4"]) + "\n".join(md) + "\n"
    with open(os.path.join(outdir, "stage_counts_%s.md" % case), "w") as f:
        f.write(text)
    print(text)
    tot = [sum(stage_counts(Sc, case, ci, None)[0] for ci in range(3)), sum(stage_counts(Sg, case, ci, None)[0] for ci in range(3))]
    ref = [stage_counts(Sc, case, None, None)[0], stage_counts(Sg, case, None, None)[0]]
    print("[check] %s: categories add up to the case sample  CC3 %s (case %s)  G4 %s (case %s)  -> %s" %
          (case, fmt(tot[0]), fmt(ref[0]), fmt(tot[1]), fmt(ref[1]), "OK" if tot == ref else "MISMATCH"))


# ----------------------------------------------------------------------------------------------
# 2. cross-tab and region table (CC3 file)
# ----------------------------------------------------------------------------------------------
def crosstab_output(Sc, case, outdir):
    m = Sc["case"][case]
    tab = np.zeros((3, 3), dtype=np.int64)
    np.add.at(tab, (Sc["cat_flag"][m], Sc["cat_cut"][m]), 1)
    tot = tab.sum()
    diag = int(np.trace(tab))
    print("\nCROSS-TAB in the CC3 file, %s: rows = class from the FLAG, columns = class from the G4 CUT (E_mc>=10 GeV), pi0 counts" % case)
    print("%-26s" % "" + "".join("%-26s" % ("cut: " + c) for c in CATEGORIES) + "row total")
    rows = []
    for i, c in enumerate(CATEGORIES):
        print("%-26s" % ("flag: " + c) + "".join("%-26s" % fmt(tab[i, j]) for j in range(3)) + fmt(tab[i].sum()))
        rows.append([case, c] + [int(x) for x in tab[i]] + [int(tab[i].sum())])
    print("agreement (diagonal share) = %.3f   (%s of %s pi0s);  flag-unknown pi0s (fast = -1): %d" % (diag / tot if tot else float("nan"), fmt(diag), fmt(tot), Sc["n_unknown_flag"]))
    write_csv(os.path.join(outdir, "crosstab_%s.csv" % case), ["case", "flag_class"] + ["cut_" + c for c in CATEGORIES] + ["row_total"], rows)


def region_output(Sc, Sg, outdir):
    rows = []
    print("\nPHOTON LEVEL: flagged fraction of generator photons with E_mc >= 10 GeV, inside / outside the barrel acceptance")
    for S in (Sc, Sg):
        a = S["a"]
        for label, want in (("inside barrel acceptance", True), ("outside barrel acceptance", False)):
            n = k = u = 0
            for j in (0, 1):
                sel = (a["mc_ph%d_gen" % j] == 1) & (a["mc_ph%d_E" % j] >= E_PHOTON_MIN)
                inb = photon_in_barrel(a["mc_ph%d_theta" % j], a["mc_ph%d_phi" % j])
                sel = sel & (inb if want else ~inb)
                n += int(sel.sum())
                k += int((a["mc_ph%d_fast" % j][sel] == 1).sum())
                u += int((a["mc_ph%d_fast" % j][sel] < 0).sum())
            print("  %-4s %-26s photons %10s  flagged %10s  fraction %.3f  unknown %d" % (S["tag"], label, fmt(n), fmt(k), (k / n if n else float("nan")), u))
            rows.append([S["tag"], label, n, k, (k / n if n else float("nan")), u])
    write_csv(os.path.join(outdir, "region_flag_fraction.csv"), ["file", "region", "photons_E>=10", "flagged", "fraction", "unknown"], rows)


# ----------------------------------------------------------------------------------------------
# figures: shared helpers
# ----------------------------------------------------------------------------------------------
def inputs_text(args):
    return "%s, %s" % (os.path.basename(args.g4_path), os.path.basename(args.cc3_path))


def caption_common(args, case, group=None, extra=""):
    s = ("Population: true pi0s with two photon daughters, both generator photons (genStatus 1). Case: %s. "
         "Categories: CC3 from %s; G4 from %s. A true photon is matched to the reconstructed particle (PFO) taking the largest part of its energy, "
         "if at least %.0f%% of that PFO's energy comes from the photon. Finder window: unfitted two-PFO mass within %.1f +- %.0f MeV. "
         "Energy axis: true pi0 energy, bins 5-10, 10-20, 20-30, 30-50, 50-125 GeV (geometric centres 7.1, 14.1, 24.5, 38.7, 79.1 GeV). "
         "A panel is drawn only with at least %d pairs in both files. Inputs: %s. "
         % (CASE_TEXT[case], CATEGORY_SOURCE["CC3"], CATEGORY_SOURCE["G4"], 100 * args.floor, 1000 * M_PI0, 1000 * args.maxdelta,
            args.min_pairs, inputs_text(args)))
    if group:
        s += "Pairs: %s. " % GROUP_TEXT[group]
    return s + extra


def place_caption(fig, caption, width, top=0.93):
    t = fig.text(0.01, 0.005, textwrap.fill(caption, width=width), fontsize=8, va="bottom", ha="left")
    fig.canvas.draw()
    h = t.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    return h


def nice_range(vals, step, qlo=0.5, qhi=99.5):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 20:
        return None
    lo, hi = np.percentile(vals, [qlo, qhi])
    lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
    if hi <= lo:
        hi = lo + step
    return lo, hi


def finite(v):
    v = np.asarray(v, dtype=float)
    return v[np.isfinite(v)]


# ----------------------------------------------------------------------------------------------
# 3. histograms of r and dtheta
# ----------------------------------------------------------------------------------------------
QUANT = {"r": dict(label="theta_reco / theta_true", step=0.05, file="ratio", title="Opening-angle ratio reco / true",
                   unit=""),
         "dtheta": dict(label="theta_reco - theta_true [mrad]", step=0.5, file="dtheta", title="Opening-angle difference reco - true",
                        unit=" mrad")}


def hist_cell(vals, edges):
    c, _ = np.histogram(vals, bins=edges)
    return c, int(c.sum()), int(len(vals) - c.sum())


def draw_distribution_figure(kind, group, case, Sc, Sg, args, outdir, summary_rows):
    q = QUANT[kind]
    fig = plt.figure(figsize=(21, 15))
    gs = fig.add_gridspec(6, 5, height_ratios=[3, 1] * 3)
    ranges = {}
    for b in DRAW_BINS:
        pooled = np.concatenate([finite(S[kind][cell_mask(S, case, None, b) & S[GROUP_KEY[group]]]) for S in (Sg, Sc)])
        ranges[b] = nice_range(pooled, q["step"])
    for ci, cname in enumerate(CATEGORIES):
        for bi, b in enumerate(DRAW_BINS):
            axh, axr = fig.add_subplot(gs[2 * ci, bi]), fig.add_subplot(gs[2 * ci + 1, bi])
            vg = finite(Sg[kind][cell_mask(Sg, case, ci, b) & Sg[GROUP_KEY[group]]])
            vc = finite(Sc[kind][cell_mask(Sc, case, ci, b) & Sc[GROUP_KEY[group]]])
            axh.set_title("%s\ntrue pi0 energy %s GeV" % (cname, E_NAMES[b]), fontsize=9)
            if ranges[b] is None or len(vg) < args.min_pairs or len(vc) < args.min_pairs:
                axh.text(0.5, 0.5, "fewer than %d pairs\nCC3 N=%d  G4 N=%d" % (args.min_pairs, len(vc), len(vg)), ha="center", va="center",
                         transform=axh.transAxes, fontsize=9)
                axh.set_xticks([]); axh.set_yticks([]); axr.axis("off")
                continue
            edges = np.linspace(ranges[b][0], ranges[b][1], args.nbins + 1)
            width = edges[1] - edges[0]
            cg, ng, bg = hist_cell(vg, edges)
            cc, nc, bc = hist_cell(vc, edges)
            for c, n, col, ls, lab, nn, bey in ((cg, ng, G4_COL, "-", "Geant4", len(vg), bg), (cc, nc, CC3_COL, "--", "CC3", len(vc), bc)):
                dens = c / (n * width) if n > 0 else np.zeros_like(c, float)
                axh.step(edges, np.append(dens, dens[-1]), where="post", color=col, ls=ls, lw=1.6, label="%s (N=%d; %d beyond axis)" % (lab, nn, bey))
            axh.legend(frameon=False, fontsize=7, loc="upper right")
            axh.spines["top"].set_visible(False); axh.spines["right"].set_visible(False)
            if bi == 0:
                axh.set_ylabel("pairs / unit " + ("ratio" if kind == "r" else "mrad"), fontsize=8)
            centres = 0.5 * (edges[1:] + edges[:-1])
            ok = (cg >= args.min_bin) & (cc >= args.min_bin) & (ng > 0) & (nc > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (cc / nc) / (cg / ng)
                err = ratio * np.sqrt(1.0 / cc + 1.0 / cg)
            show = ok & (ratio >= 0.4) & (ratio <= 1.6)
            axr.errorbar(centres[show], ratio[show], yerr=err[show], fmt="o", color="black", ms=3, ecolor="gray", lw=1)
            axr.axhline(1.0, color="gray", lw=1)
            axr.set_ylim(0.4, 1.6); axr.set_xlim(edges[0], edges[-1])
            axr.set_xlabel(q["label"], fontsize=8)
            if bi == 0:
                axr.set_ylabel("CC3 / G4", fontsize=8)
            axr.spines["top"].set_visible(False); axr.spines["right"].set_visible(False)
            axh.set_xlim(edges[0], edges[-1])
    ttl = "%s: %s\n%s" % (q["title"], GROUP_TEXT[group], CASE_TEXT[case])
    fig.suptitle(ttl, fontsize=12)
    cap = caption_common(args, case, group, "Histograms are normalised to unit area over the drawn range, so the ratio compares SHAPES. "
                         "Ratio error: independent Poisson on the two bin counts (approximate, the files share generator events); "
                         "ratio points need at least %d entries in both files; points outside 0.4-1.6 are not shown." % args.min_bin)
    h = place_caption(fig, cap, 190)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.91, bottom=h + 0.04, hspace=0.55, wspace=0.22)
    fig.savefig(os.path.join(outdir, "%s_%s_%s.png" % (q["file"], group, case)), dpi=args.dpi)
    plt.close(fig)


def angle_summary(Sc, Sg, case, args, outdir):
    rows = []
    for group in ("resolved", "found"):
        key = GROUP_KEY[group]
        for ci, cname in enumerate(CATEGORIES):
            for b in DRAW_BINS + ["5-125"]:
                bb = list(DRAW_BINS) if b == "5-125" else b
                name = "5-125 (drawn bins)" if b == "5-125" else E_NAMES[b]
                out = [case, group, cname, name]
                for S in (Sc, Sg):
                    m = cell_mask(S, case, ci, bb) & S[key]
                    r, d = finite(S["r"][m]), finite(S["dtheta"][m])
                    out += [len(r)] + ([float(np.median(r)), float(s68(r)), float(np.median(d)), float(s68(d))] if len(r) else [float("nan")] * 4)
                rows.append(out)
    hdr = ["case", "pairs", "category", "true_E_bin"]
    for t in ("CC3", "G4"):
        hdr += ["%s_n" % t, "%s_median_r" % t, "%s_s68_r" % t, "%s_median_dtheta_mrad" % t, "%s_s68_dtheta_mrad" % t]
    write_csv(os.path.join(outdir, "angle_summary_%s.csv" % case), hdr, rows)


# ----------------------------------------------------------------------------------------------
# 4. found fraction vs pair mass (plot G by category)
# ----------------------------------------------------------------------------------------------
def draw_found_fraction_figure(case, Sc, Sg, args, outdir):
    edges = np.arange(0.0, 300.0 + args.mass_bin, float(args.mass_bin))
    centres = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = plt.subplots(3, 5, figsize=(21, 12), sharey=True)
    csv_rows = []
    for ci, cname in enumerate(CATEGORIES):
        for bi, b in enumerate(DRAW_BINS):
            ax = axes[ci][bi]
            ax.set_title("%s\ntrue pi0 energy %s GeV" % (cname, E_NAMES[b]), fontsize=9)
            data = {}
            for S in (Sg, Sc):
                m = cell_mask(S, case, ci, b) & S["res_E"]
                mass = S["mass"][m] * 1000.0
                fnd = S["found"][m]
                good = np.isfinite(mass)
                data[S["tag"]] = (mass[good], fnd[good])
            ng, nc = len(data["G4"][0]), len(data["CC3"][0])
            for tag in ("G4", "CC3"):
                mass, fnd = data[tag]
                tot, _ = np.histogram(mass, bins=edges)
                k, _ = np.histogram(mass[fnd], bins=edges)
                for i in range(len(centres)):
                    csv_rows.append([case, cname, E_NAMES[b], tag, edges[i], edges[i + 1], int(tot[i]), int(k[i])])
            ax.axvline(1000 * (M_PI0 - args.maxdelta), color="gray", ls=":", lw=1)
            ax.axvline(1000 * (M_PI0 + args.maxdelta), color="gray", ls=":", lw=1)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if ng < args.min_pairs or nc < args.min_pairs:
                ax.text(0.5, 0.5, "fewer than %d pairs\nCC3 N=%d  G4 N=%d" % (args.min_pairs, nc, ng), ha="center", va="center", transform=ax.transAxes, fontsize=9)
                continue
            for tag, col, mk in (("G4", G4_COL, "o"), ("CC3", CC3_COL, "s")):
                mass, fnd = data[tag]
                tot, _ = np.histogram(mass, bins=edges)
                k, _ = np.histogram(mass[fnd], bins=edges)
                ok = tot >= args.min_bin
                frac = np.divide(k, tot, out=np.zeros(len(tot)), where=tot > 0)
                err = np.array([binom_err(k[i], tot[i]) for i in range(len(tot))])
                ax.errorbar(centres[ok], frac[ok], yerr=err[ok], fmt=mk, ms=4, color=col, lw=1, label="%s (N=%d)" % ("Geant4" if tag == "G4" else "CC3", len(mass)))
            ax.legend(frameon=False, fontsize=7, loc="upper right")
            ax.set_ylim(0, 1.05)
            if bi == 0:
                ax.set_ylabel("found / pairs in the mass bin", fontsize=8)
            if ci == 2:
                ax.set_xlabel("unfitted pair mass [MeV]", fontsize=8)
    fig.suptitle("How the finder treats pairs: fraction kept as a candidate vs unfitted pair mass\n%s" % CASE_TEXT[case], fontsize=12)
    cap = caption_common(args, case, None, "Pairs: two separate photon PFOs, both PFO energies >= %.1f GeV (the pairs the finder could keep). "
                         "Pair mass: invariant mass of the two PFOs, unfitted. Found = the finder made a candidate from exactly these two PFOs. "
                         "Dotted lines: edges of the finder window you gave; the found fraction is 0 outside it by construction. "
                         "Points need at least %d pairs in the mass bin; error bars binomial; mass bins %g MeV wide." % (E_CUT, args.min_bin, args.mass_bin))
    h = place_caption(fig, cap, 190)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=h + 0.06, hspace=0.45, wspace=0.12)
    fig.savefig(os.path.join(outdir, "G_found_fraction_vs_mass_%s.png" % case), dpi=args.dpi)
    plt.close(fig)
    write_csv(os.path.join(outdir, "found_fraction_vs_mass_%s.csv" % case),
              ["case", "category", "true_E_bin", "file", "mass_lo_MeV", "mass_hi_MeV", "pairs", "found"], csv_rows)


# ----------------------------------------------------------------------------------------------
# global self-checks
# ----------------------------------------------------------------------------------------------
def global_checks(Sc, Sg, args):
    print("\nSELF-CHECKS")
    for S in (Sc, Sg):
        a = S["a"]
        n_fl = int(((a["mc_ph0_fast"] == 1) | (a["mc_ph1_fast"] == 1))[S["gen"]].sum())
        print("  %-4s generator pi0s %s | pi0s with a flagged photon %s%s | flag-unknown pi0s %d | negative m^2 in resolved pairs %d" %
              (S["tag"], fmt(S["gen"].sum()), fmt(n_fl), "  (G4 file must have 0)" if S["tag"] == "G4" else "", S["n_unknown_flag"], S["neg_m2"]))
        fo = S["found"] & np.isfinite(S["mass"]) & ((S["mass"] < M_PI0 - args.maxdelta) | (S["mass"] > M_PI0 + args.maxdelta))
        print("  %-4s found pairs outside the window you gave (%.1f +- %.0f MeV): %d  (must be 0 if --maxdelta matches the finder run)" %
              (S["tag"], 1000 * M_PI0, 1000 * args.maxdelta, int(fo.sum())))
    g, c = Sg["bin_true"][Sg["gen"]], Sc["bin_true"][Sc["gen"]]
    diffs = [int((c == b).sum() - (g == b).sum()) for b in DRAW_BINS]
    print("  same generator pi0s in both files? CC3 - G4 per true-energy bin %s (the log has differences of 0-3)" % diffs)
    exp = LOG_EXPECT.get(args.suffix)
    if exp and abs(args.floor - 0.5) < 1e-9:
        for S in (Sg, Sc):
            n_true, n_found = int(S["gen"].sum()), int((S["gen"] & S["found"]).sum())
            e = exp[S["tag"]]
            print("  %-4s vs findings log (floor 0.5): N_true %s (log %s), found %s (log %s) -> %s" %
                  (S["tag"], fmt(n_true), fmt(e[0]), fmt(n_found), fmt(e[1]), "OK" if (n_true, n_found) == e else "DIFFERENT"))


# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Reco-side category study, G4 vs CC3. Run with no arguments for the 0.04 tables.")
    ap.add_argument("--indir", default=REC_DIR)
    ap.add_argument("--suffix", default="full")
    ap.add_argument("--g4", default=None, help="explicit path of the G4 table (overrides indir/suffix)")
    ap.add_argument("--cc3", default=None, help="explicit path of the CC3 table (overrides indir/suffix)")
    ap.add_argument("--maxdelta", type=float, default=None, help="finder window half-width in GeV; default 0.1 if 'dmgg0.1' is in the suffix, else 0.04")
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--case", choices=["full", "barrel", "both"], default="both")
    ap.add_argument("--pairs", nargs="+", choices=["resolved", "found"], default=["resolved", "found"])
    ap.add_argument("--min-pairs", type=int, default=100)
    ap.add_argument("--min-bin", type=int, default=10)
    ap.add_argument("--nbins", type=int, default=30)
    ap.add_argument("--mass-bin", type=float, default=20.0)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args(argv)
    if args.maxdelta is None:
        args.maxdelta = 0.1 if "dmgg0.1" in args.suffix else 0.04
    args.g4_path = args.g4 or os.path.join(args.indir, "pi0-tables-g4-%s.npz" % args.suffix)
    args.cc3_path = args.cc3 or os.path.join(args.indir, "pi0-tables-cc3-%s.npz" % args.suffix)
    return args


def main(argv=None):
    args = parse_args(argv)
    outdir = args.outdir or os.path.join(PLOT_DIR, "category-study-%s" % args.suffix)
    os.makedirs(outdir, exist_ok=True)
    sys.stdout = Tee(os.path.join(outdir, "summary.txt"))
    try:
        print("category_study.py | inputs: %s , %s | finder window +-%.3f GeV | floor %.2f | min pairs %d | outdir %s" %
              (args.g4_path, args.cc3_path, args.maxdelta, args.floor, args.min_pairs, outdir))
        Sg = build_state(load_tables(args.g4_path), "G4", args.floor, args.maxdelta)
        Sc = build_state(load_tables(args.cc3_path), "CC3", args.floor, args.maxdelta)
        global_checks(Sc, Sg, args)
        region_output(Sc, Sg, outdir)
        cases = ["full", "barrel"] if args.case == "both" else [args.case]
        for case in cases:
            print("\n" + "=" * 100 + "\nCASE: %s" % CASE_TEXT[case])
            counts_outputs(Sc, Sg, case, outdir)
            crosstab_output(Sc, case, outdir)
            angle_summary(Sc, Sg, case, args, outdir)
            draw_found_fraction_figure(case, Sc, Sg, args, outdir)
            for group in args.pairs:
                for kind in ("r", "dtheta"):
                    draw_distribution_figure(kind, group, case, Sc, Sg, args, outdir, None)
        print("\nDone. Output folder: %s" % outdir)
    finally:
        sys.stdout = sys.__stdout__
    return outdir


if __name__ == "__main__":
    main()
