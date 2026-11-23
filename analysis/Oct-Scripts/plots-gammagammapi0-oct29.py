#!/usr/bin/env python3
"""
plot_pi0_october29.py -- ONE pass over the two NPZ files (G4 and CC3) written by make_pi0_tables.py -> all plots + CSVs.

RUN WITH NO ARGUMENTS for the 0.04 files:   python plot_pi0_october29.py
0.1 files (after the maker has produced them):
    python plot_pi0_october29.py --suffix full-dmgg0.1 --maxdelta 0.1
Quick test on the 1k files:   python plot_pi0_october29.py --suffix first1000

PLOTS (one PNG each, in the output folder):
  A  efficiency vs TRUE pi0 energy: found / true pi0s, G4 and CC3, plus CC3/G4 ratio.
  B  where in the chain CC3 falls behind: CC3/G4 count ratio after each cumulative stage, vs TRUE pi0 energy.
  D  reco opening angle distribution, one column per energy bin, G4 vs CC3 and CC3/G4 per angle bin.
     two versions: "finder" (pairs the finder turned into a candidate) and "resolved" (all pairs with two separate photon PFOs).
  E  angle response = reco opening angle / true opening angle: median, 68% half-width, and CC3/G4 of the half-width.
     two versions in one figure: resolved pairs (filled) and finder pairs (open).
CSVs: stage_counts.csv (A, B), angle_response.csv (E), summary.txt (everything printed, including self-checks).

DEFINITIONS (same as analyse_pi0_tables.py -- see npz-tables-reference.md):
  population  : true pi0 with exactly two photon daughters, both genStatus 1 (generator pi0s). No theta/phi/fast-sim cuts.
  matched PFO : best PFO of a photon (ranked by cw*E_pfo) counts if cw >= floor (default 0.5).
  resolved    : both photons matched, to two DIFFERENT PFOs, both of type photon (PDG 22).
  found       : a finder candidate exists whose two daughters are exactly these two PFOs.
  stages      : resolved -> + both PFO E >= 0.5 GeV -> + unfitted pair mass inside the finder window -> found.
  window      : 0.1349766 +- maxdelta GeV on the unfitted mass of the two PFOs (maxdelta = MaxDeltaMgg of the finder run).
  reco angle  : angle between the momentum vectors of the two best PFOs (the candidate's fitted momentum is only a pair sum).
  energy axis : --energy-axis true (default) bins by true pi0 energy; reco bins by the sum of the two PFO energies.
                A and B are always in true energy (they need the truth denominator).
STATISTICS: efficiencies binomial; CC3/G4 ratios of counts independent Poisson; efficiency-ratio error from the two binomials.
  G4 and CC3 share generator events, so ratio errors are APPROXIMATE. Medians / half-widths: rough errors (1.2533*s68/sqrt(N),
  s68/sqrt(2N)). No systematics. Points with fewer than --min-count entries are not drawn in ratio panels.
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
from matplotlib.ticker import NullLocator, ScalarFormatter

REC_DIR = "/data/dust/user/alimuham/thesis/reco"
PLOT_DIR = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots"
BASE_NAME = "gammagammapi0-october-29"
E_EDGES = [5.0, 10.0, 20.0, 30.0, 50.0, 125.0]          # lower edge inclusive
E_NAMES = ["<5", "5-10", "10-20", "20-30", "30-50", "50-125", ">=125"]
DRAW_BINS = [1, 2, 3, 4, 5]                               # 5-10 ... 50-125 GeV (the open-ended bins are in the CSV only)
E_CUT = 0.5                                               # GeV, finder photon energy cut
M_PI0 = 0.1349766                                         # GeV
EPS = 1e-6
G4_COL, CC3_COL = "#2166ac", "#d6604d"
STAGES = [("resolved", "two separate photon PFOs"),
          ("res_E", "+ both PFO energies >= 0.5 GeV"),
          ("res_E_window", "+ pair mass inside the finder window"),
          ("found", "found by the finder")]
MIN_N_E = 20                                              # minimum pairs for a median / half-width point
LOG_EXPECT = {"G4": dict(N_true=176406, found=60907), "CC3": dict(N_true=176402, found=58038)}   # findings log, 0.04, floor 0.5

REQUIRED = ["mc_loop_evt", "mc_pi0_E", "mc_ph0_gen", "mc_ph1_gen", "mc_true_open_angle",
            "cand_loop_evt", "cand_d0_pfo_idx", "cand_d1_pfo_idx"]
for _k in (0, 1):
    for _c in ("pfo_idx", "cw", "pdg", "E", "px", "py", "pz"):
        REQUIRED.append("mc_ph%d_b0_%s" % (_k, _c))


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def load_tables(path, tag):
    z = np.load(path)
    a = {k: z[k] for k in z.files if k.startswith(("mc_", "cand_"))}
    missing = [c for c in REQUIRED if c not in a]
    if missing:
        raise KeyError("%s: columns missing in %s: %s" % (tag, path, missing))
    return dict(tag=tag, path=path, a=a)


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


def ratio_err(a, b):
    """a/b with independent Poisson counts a and b."""
    return a / b * math.sqrt(1.0 / a + 1.0 / b) if a > 0 and b > 0 else float("nan")


def eff_ratio_err(kc, nc, kg, ng):
    """(kc/nc)/(kg/ng) with the two binomial errors added in quadrature (approximate)."""
    if min(kc, nc, kg, ng) <= 0:
        return float("nan"), float("nan")
    ec, eg = kc / nc, kg / ng
    sc, sg = binom_err(kc, nc), binom_err(kg, ng)
    r = ec / eg
    return r, r * math.sqrt((sc / ec) ** 2 + (sg / eg) ** 2)


def centers_and_edges():
    lo = np.array([E_EDGES[b - 1] for b in DRAW_BINS])
    hi = np.array([E_EDGES[b] for b in DRAW_BINS])
    return np.sqrt(lo * hi), lo, hi


def set_energy_axis(ax, label=None):
    ax.set_xscale("log")
    ax.set_xlim(4.5, 140)
    ax.set_xticks(E_EDGES)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    if label:
        ax.set_xlabel(label)


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def nice_ceil(x):
    step = 5.0 if x >= 20 else 2.0 if x >= 8 else 1.0
    return max(step, math.ceil(x / step) * step)


# ----------------------------------------------------------------------------------------------
# per-file state: resolved / energy cut / window / found, reco angle
# ----------------------------------------------------------------------------------------------
def candidate_keys(a):
    ok = (a["cand_d0_pfo_idx"] >= 0) & (a["cand_d1_pfo_idx"] >= 0)
    return pair_key(a["cand_loop_evt"][ok], a["cand_d0_pfo_idx"][ok], a["cand_d1_pfo_idx"][ok])


def pair_state(a, floor, m_lo, m_hi):
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
        angle_reco = np.arctan2(np.linalg.norm(np.cross(v0, v1), axis=1), np.sum(v0 * v1, axis=1))
        E_ok = res & (e0 >= E_CUT) & (e1 >= E_CUT)
        M_ok = res & (mass >= m_lo) & (mass <= m_hi)
    found = np.zeros(ia.size, dtype=bool)
    idx = np.where(res)[0]
    if idx.size:
        found[idx] = np.isin(pair_key(a["mc_loop_evt"][idx], ia[idx], ib[idx]), candidate_keys(a))
    return dict(res=res, resolved=res, res_E=E_ok, res_E_window=E_ok & M_ok, found=found, mass=mass,
                angle_reco=angle_reco, E_reco_pair=e0 + e1)


def stage_counts(mask, bins):
    out = np.zeros(8, dtype=np.int64)
    ok = mask & (bins >= 0)
    out[:7] = np.bincount(bins[ok], minlength=7)[:7]
    out[7] = out[:7].sum()
    return out


def build_file(d, floor, maxdelta, energy_axis):
    a = d["a"]
    gen = (a["mc_ph0_gen"] == 1) & (a["mc_ph1_gen"] == 1)
    st = pair_state(a, floor, M_PI0 - maxdelta, M_PI0 + maxdelta)
    bin_true = bin_index(a["mc_pi0_E"])
    bin_reco = bin_index(st["E_reco_pair"])
    counts = {"N_true": stage_counts(gen, bin_true)}
    for name, _ in STAGES:
        counts[name] = stage_counts(gen & st[name], bin_true)
    return dict(tag=d["tag"], a=a, gen=gen, st=st, bin_true=bin_true, bin_reco=bin_reco,
                bin_axis=(bin_true if energy_axis == "true" else bin_reco), counts=counts)


# ----------------------------------------------------------------------------------------------
# captions and figure frame
# ----------------------------------------------------------------------------------------------
def caption_common(args, extra=""):
    s = ("Population: true pi0s with two photon daughters, both generator photons (genStatus 1); no theta/phi or fast-sim cuts, "
         "all detector regions. A true photon is matched to the reconstructed particle (PFO) taking the largest part of its energy, "
         "if at least %.0f%% of that PFO's energy comes from the photon. Finder window: unfitted two-PFO mass within "
         "%.1f +- %.0f MeV. Inputs: pi0-tables-{g4,cc3}-%s.npz. " % (100 * args["floor"], 1000 * M_PI0, 1000 * args["maxdelta"], args["suffix"]))
    return s + extra


def place_caption(fig, caption, width, top=0.97):
    """Write the caption at the bottom, measure its height, and leave exactly that much room for the axes."""
    t = fig.text(0.01, 0.01, textwrap.fill(caption, width=width), fontsize=7, va="bottom", ha="left")
    fig.canvas.draw()
    h = t.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    fig.tight_layout(rect=(0, h + 0.025, 1, top))


def finish_fig(fig, caption, width):
    place_caption(fig, caption, width)


# ----------------------------------------------------------------------------------------------
# plot A: efficiency
# ----------------------------------------------------------------------------------------------
def plot_A(SG, SC, args, outdir):
    cen, lo, hi = centers_and_edges()
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(7.5, 7.0), gridspec_kw={"height_ratios": [3, 1.5]})
    ratios = []
    for S, col, lab, mk in ((SG, G4_COL, "Geant4", "o"), (SC, CC3_COL, "CC3", "s")):
        n = np.array([S["counts"]["N_true"][b] for b in DRAW_BINS], float)
        k = np.array([S["counts"]["found"][b] for b in DRAW_BINS], float)
        eff = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
        err = np.array([binom_err(k[i], n[i]) for i in range(len(n))])
        tot = S["counts"]["found"][7] / S["counts"]["N_true"][7]
        ax.errorbar(cen, eff, yerr=err, xerr=[cen - lo, hi - cen], fmt=mk, color=col, ms=5, capsize=0,
                    label="%s (all energies: %.3f)" % (lab, tot))
    for i, b in enumerate(DRAW_BINS):
        r, e = eff_ratio_err(SC["counts"]["found"][b], SC["counts"]["N_true"][b], SG["counts"]["found"][b], SG["counts"]["N_true"][b])
        ratios.append((r, e))
    r = np.array([x[0] for x in ratios])
    e = np.array([x[1] for x in ratios])
    axr.errorbar(cen, r, yerr=e, xerr=[cen - lo, hi - cen], fmt="o", color="black", ms=4, ecolor="gray")
    axr.axhline(1.0, color="gray", lw=1)
    rt, et = eff_ratio_err(SC["counts"]["found"][7], SC["counts"]["N_true"][7], SG["counts"]["found"][7], SG["counts"]["N_true"][7])
    axr.axhline(rt, color=CC3_COL, ls="--", lw=1, label="all energies %.3f +- %.3f" % (rt, et))
    fin = np.isfinite(r)
    lo_y = min(0.8, np.nanmin(r[fin] - e[fin]) - 0.02) if fin.any() else 0.8
    hi_y = max(1.05, np.nanmax(r[fin] + e[fin]) + 0.02) if fin.any() else 1.05
    axr.set_ylim(lo_y, hi_y)
    ax.set_ylabel("found by the finder / true pi0s")
    axr.set_ylabel("CC3 / G4")
    set_energy_axis(axr, "true pi0 energy [GeV]")
    ax.legend(frameon=False, fontsize=9)
    axr.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("What fraction of generator pi0s does the finder deliver, Geant4 vs CC3?", fontsize=10)
    clean(ax); clean(axr)
    cap = caption_common(args, "Found = the finder made a candidate from exactly the two matched PFOs. "
                               "Error bars: binomial; ratio error from the two binomials (approximate, the files share generator events). "
                               "Horizontal bars show the bin width. Bins below 5 GeV and above 125 GeV are not drawn (see stage_counts.csv).")
    finish_fig(fig, cap, 105)
    fig.savefig(os.path.join(outdir, "A_efficiency_vs_true_energy.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# plot B: stage ratios
# ----------------------------------------------------------------------------------------------
def plot_B(SG, SC, args, outdir):
    cen, lo, hi = centers_and_edges()
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    series = [("N_true", "all generator pi0s", "0.55", "x", 0.93)] + \
             [(s, lab, c, m, f) for (s, lab), c, m, f in zip(STAGES, ("#1b9e77", "#7570b3", "#e6ab02", "black"), ("o", "s", "^", "D"), (0.97, 1.0, 1.03, 1.07))]
    allv = []
    for key, lab, col, mk, off in series:
        r = np.full(len(DRAW_BINS), np.nan)
        e = np.full(len(DRAW_BINS), np.nan)
        for i, b in enumerate(DRAW_BINS):
            g, c = SG["counts"][key][b], SC["counts"][key][b]
            if g > 0 and c > 0:
                r[i], e[i] = c / g, ratio_err(c, g)
        ax.errorbar(cen * off, r, yerr=e, fmt=mk + "-", color=col, ms=5, lw=1, label=lab)
        allv += list(r[np.isfinite(r)] - e[np.isfinite(e)]) + list(r[np.isfinite(r)] + e[np.isfinite(e)])
    ax.axhline(1.0, color="gray", lw=1)
    ax.set_ylim(min(0.75, min(allv) - 0.02) if allv else 0.75, max(1.08, max(allv) + 0.02) if allv else 1.08)
    set_energy_axis(ax, "true pi0 energy [GeV]")
    ax.set_ylabel("CC3 / G4 (number of pi0s still in the chain)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("At which stage of the chain does CC3 fall behind Geant4?", fontsize=10)
    clean(ax)
    cap = caption_common(args, "Each line is the CC3/G4 ratio of the number of true pi0s that pass that stage AND all earlier stages. "
                               "Error bars: independent Poisson (approximate, shared generator events). "
                               "Points of different stages are shifted slightly sideways for readability.")
    finish_fig(fig, cap, 105)
    fig.savefig(os.path.join(outdir, "B_stage_ratios_vs_true_energy.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# plot D: opening-angle distributions
# ----------------------------------------------------------------------------------------------
def select_pairs(S, version, b):
    m = S["gen"] & S["st"][version] & (S["bin_axis"] == b) & np.isfinite(S["st"]["angle_reco"])
    return S["st"]["angle_reco"][m] * 1000.0                      # mrad


def plot_D(SG, SC, args, outdir, version):
    ename = "true pi0 energy" if args["energy_axis"] == "true" else "reco pair energy"
    fig, axes = plt.subplots(2, len(DRAW_BINS), figsize=(4.0 * len(DRAW_BINS), 7.2), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.4]})
    for j, b in enumerate(DRAW_BINS):
        top, bot = axes[0, j], axes[1, j]
        ag, ac = select_pairs(SG, version, b), select_pairs(SC, version, b)
        top.set_title("%s %s GeV" % (ename, E_NAMES[b]), fontsize=9)
        if ag.size == 0 and ac.size == 0:
            top.text(0.5, 0.5, "no pairs", transform=top.transAxes, ha="center")
            continue
        hi_x = nice_ceil(max([np.percentile(x, 99.5) for x in (ag, ac) if x.size]))
        edges = np.linspace(0.0, hi_x, 31)
        w = edges[1] - edges[0]
        hg, _ = np.histogram(ag, bins=edges)
        hc, _ = np.histogram(ac, bins=edges)
        xs = np.repeat(edges, 2)[1:-1]
        for h, col, ls, lab, a_ in ((hg, G4_COL, "-", "Geant4", ag), (hc, CC3_COL, "--", "CC3", ac)):
            over = int((a_ >= hi_x).sum())
            top.plot(xs, np.repeat(h / w, 2), color=col, ls=ls, lw=1.8,
                     label="%s (N=%d%s)" % (lab, a_.size, "; %d beyond axis" % over if over else ""))
        top.legend(frameon=False, fontsize=7)
        top.set_ylabel("pairs / mrad" if j == 0 else "")
        ok = (hg >= args["min_count"]) & (hc >= args["min_count"])
        cen = 0.5 * (edges[1:] + edges[:-1])
        r = hc[ok] / hg[ok]
        e = r * np.sqrt(1.0 / hc[ok] + 1.0 / hg[ok])
        bot.errorbar(cen[ok], r, yerr=e, fmt="o", color="black", ms=3, ecolor="gray", lw=1)
        bot.axhline(1.0, color="gray", lw=1)
        if ag.size and ac.size:
            bot.axhline(ac.size / ag.size, color=CC3_COL, ls="--", lw=1, label="all pairs in this column: %.3f" % (ac.size / ag.size))
            bot.legend(frameon=False, fontsize=7, loc="lower right")
        bot.set_ylim(0.4, 1.6)
        bot.set_xlim(0, hi_x)
        bot.set_xlabel("opening angle of the two reco photons [mrad]", fontsize=8)
        bot.set_ylabel("CC3 / G4" if j == 0 else "")
        clean(top); clean(bot)
    if version == "finder":
        ttl = "Do CC3 and Geant4 differ in the opening angle of the pairs the finder keeps?"
        what = "Pairs: the finder made a candidate from exactly these two PFOs."
    else:
        ttl = "Do CC3 and Geant4 differ in the opening angle of all pairs with two separate photon PFOs?"
        what = "Pairs: every true pi0 whose two photons each have their own photon PFO, whether or not the finder kept the pair."
    fig.suptitle(ttl, fontsize=11)
    cap = caption_common(args, what + " Energy axis: %s. Opening angle from the momenta of the two PFOs. "
                               "Error bars on the ratio: independent Poisson (approximate); ratio points need at least %d pairs in both files; "
                               "points outside 0.4-1.6 are not shown. Pairs beyond the x range are counted in N but not drawn."
                         % (ename + (" (sum of the two PFO energies)" if args["energy_axis"] == "reco" else ""), args["min_count"]))
    place_caption(fig, cap, 190, top=0.95)
    fig.savefig(os.path.join(outdir, "D_opening_angle_%s_pairs.png" % version), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# plot E: angle response
# ----------------------------------------------------------------------------------------------
def angle_ratio_stats(r):
    r = r[np.isfinite(r)]
    n = r.size
    if n < MIN_N_E:
        return None
    p16, p50, p84 = np.percentile(r, [16, 50, 84])
    s68 = 0.5 * (p84 - p16)
    return dict(n=n, median=float(p50), s68=float(s68), err_med=1.2533 * s68 / math.sqrt(n), err_s68=s68 / math.sqrt(2.0 * n))


def response_values(S, version, b):
    m = S["gen"] & S["st"][version] & (S["bin_axis"] == b)
    at = S["a"]["mc_true_open_angle"].astype(float)
    m &= np.isfinite(at) & (at > 0) & np.isfinite(S["st"]["angle_reco"])
    return S["st"]["angle_reco"][m] / at[m]


def plot_E(SG, SC, args, outdir):
    cen, _, _ = centers_and_edges()
    ename = "true pi0 energy" if args["energy_axis"] == "true" else "reco pair energy [sum of the two PFO energies]"
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.8, 9.2), gridspec_kw={"height_ratios": [2, 2, 1.5]})
    am, as_, ar = axes
    rows = []
    stats = {}
    spec = [(SG, "resolved", G4_COL, "o", True, 0.94, "Geant4, all resolved pairs"),
            (SG, "found", G4_COL, "o", False, 0.98, "Geant4, finder pairs"),
            (SC, "resolved", CC3_COL, "s", True, 1.02, "CC3, all resolved pairs"),
            (SC, "found", CC3_COL, "s", False, 1.06, "CC3, finder pairs")]
    for S, ver, col, mk, filled, off, lab in spec:
        xs, med, emed, s68, es68 = [], [], [], [], []
        for i, b in enumerate(DRAW_BINS):
            st = angle_ratio_stats(response_values(S, ver, b))
            stats[(S["tag"], ver, b)] = st
            if st is None:
                continue
            xs.append(cen[i] * off); med.append(st["median"]); emed.append(st["err_med"])
            s68.append(st["s68"]); es68.append(st["err_s68"])
            rows.append([S["tag"], ver, E_NAMES[b], st["n"], st["median"], st["err_med"], st["s68"], st["err_s68"]])
        kw = dict(fmt=mk, color=col, ms=5, mfc=(col if filled else "white"), capsize=0, lw=1)
        am.errorbar(xs, med, yerr=emed, label=lab, **kw)
        as_.errorbar(xs, s68, yerr=es68, **kw)
    for ver, mk, filled, off in (("resolved", "o", True, 0.98), ("found", "o", False, 1.02)):
        xs, rr, er = [], [], []
        for i, b in enumerate(DRAW_BINS):
            g, c = stats.get(("G4", ver, b)), stats.get(("CC3", ver, b))
            if g is None or c is None:
                continue
            r = c["s68"] / g["s68"]
            xs.append(cen[i] * off); rr.append(r)
            er.append(r * math.sqrt(1.0 / (2 * c["n"]) + 1.0 / (2 * g["n"])))
        ar.errorbar(xs, rr, yerr=er, fmt=mk, color="black", ms=5, mfc=("black" if filled else "white"), lw=1,
                    label="all resolved pairs" if filled else "finder pairs")
    am.axhline(1.0, color="gray", lw=1)
    ar.axhline(1.0, color="gray", lw=1)
    am.set_ylabel("median of reco angle / true angle")
    as_.set_ylabel("68% half-width of reco angle / true angle")
    ar.set_ylabel("CC3 / G4 of the half-width")
    set_energy_axis(ar, ename + " [GeV]")
    am.legend(frameon=False, fontsize=8)
    ar.legend(frameon=False, fontsize=8)
    am.set_title("Does CC3 reconstruct the opening angle of the photon pair differently from Geant4?", fontsize=10)
    for x in axes:
        clean(x)
    cap = caption_common(args, "Reco angle: between the momenta of the two PFOs; true angle: between the two generator photons. "
                               "Filled markers: all pairs with two separate photon PFOs; open markers: pairs the finder kept. "
                               "Energy axis: %s. Points with fewer than %d pairs are not drawn. Error bars on medians (1.2533*s68/sqrt(N)) and "
                               "half-widths (s68/sqrt(2N)) are rough; ratio errors combine them (approximate, shared generator events). "
                               "Markers are shifted sideways for readability." % (ename, MIN_N_E))
    finish_fig(fig, cap, 105)
    fig.savefig(os.path.join(outdir, "E_angle_response.png"), dpi=150)
    plt.close(fig)
    return rows, stats


# ----------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------
def run(path_g4, path_cc3, outdir, floor=0.5, maxdelta=0.04, energy_axis="true", min_count=10, suffix="full"):
    os.makedirs(outdir, exist_ok=True)
    args = dict(floor=floor, maxdelta=maxdelta, energy_axis=energy_axis, min_count=min_count, suffix=suffix)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("plot_pi0_october29.py | floor %.2f | maxdelta %.3f GeV (window %.4f-%.4f GeV) | energy axis %s" %
        (floor, maxdelta, M_PI0 - maxdelta, M_PI0 + maxdelta, energy_axis))
    say("G4 : " + path_g4)
    say("CC3: " + path_cc3)
    SG = build_file(load_tables(path_g4, "G4"), floor, maxdelta, energy_axis)
    SC = build_file(load_tables(path_cc3, "CC3"), floor, maxdelta, energy_axis)

    say("\n[1] STAGE COUNTS per TRUE energy bin (generator pi0s, cumulative stages)")
    rows = []
    for S in (SG, SC):
        say("  %s" % S["tag"])
        say("    %-8s %10s %10s %10s %14s %10s" % ("bin", "N_true", "resolved", "res+E", "res+E+window", "found"))
        for b in range(8):
            nm = "all" if b == 7 else E_NAMES[b]
            v = [int(S["counts"][k][b]) for k in ("N_true", "resolved", "res_E", "res_E_window", "found")]
            say("    %-8s %10d %10d %10d %14d %10d" % ((nm,) + tuple(v)))
            rows.append([S["tag"], nm] + v)
    write_csv(os.path.join(outdir, "stage_counts.csv"), ["file", "true_E_bin", "N_true", "resolved", "res_E", "res_E_window", "found"], rows)

    say("\n[2] SELF-CHECKS")
    for S in (SG, SC):
        st = S["st"]
        bad_w = int((S["gen"] & st["found"] & ~st["res_E_window"]).sum())
        bad_e = int((S["gen"] & st["found"] & ~st["res_E"]).sum())
        say("  %s: found pairs OUTSIDE the window you gave: %d | found pairs failing the 0.5 GeV cut: %d" % (S["tag"], bad_w, bad_e))
        if bad_w > 0:
            say("      WARNING: the finder accepted pairs outside +-%.0f MeV -> check --maxdelta against the MaxDeltaMgg of this run." % (1000 * maxdelta))
        say("  %s: resolved pairs with non-finite reco angle: %d" % (S["tag"], int((st["res"] & ~np.isfinite(st["angle_reco"])).sum())))
    if suffix == "full" and abs(maxdelta - 0.04) < 1e-12 and abs(floor - 0.5) < 1e-12:
        for S in (SG, SC):
            n, k = int(S["counts"]["N_true"][7]), int(S["counts"]["found"][7])
            exp = LOG_EXPECT[S["tag"]]
            say("  %s vs findings log (0.04, floor 0.5): N_true %d (log %d), found %d (log %d) -> %s" %
                (S["tag"], n, exp["N_true"], k, exp["found"], "MATCH" if (n, k) == (exp["N_true"], exp["found"]) else "DIFFERENT"))
    else:
        say("  (comparison with the 0.04 numbers of the findings log skipped: not the 0.04/full/floor-0.5 setting)")

    plot_A(SG, SC, args, outdir)
    plot_B(SG, SC, args, outdir)
    plot_D(SG, SC, args, outdir, "found")
    plot_D(SG, SC, args, outdir, "resolved")
    rows_e, stats = plot_E(SG, SC, args, outdir)
    write_csv(os.path.join(outdir, "angle_response.csv"),
              ["file", "version", "energy_bin", "N", "median_reco_over_true", "err_median", "s68_reco_over_true", "err_s68"], rows_e)
    say("\nFiles written to " + outdir)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return dict(SG=SG, SC=SC, stats=stats)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run with no arguments: 0.04 full files, plots in the October-29 folder.")
    ap.add_argument("--indir", default=REC_DIR)
    ap.add_argument("--suffix", default="full", help="full (default), first1000, full-dmgg0.1, ...")
    ap.add_argument("--maxdelta", type=float, default=0.04, help="MaxDeltaMgg of the finder run in GeV (0.04 or 0.1)")
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--energy-axis", choices=["true", "reco"], default="true", help="for plots D and E")
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)
    outdir = a.outdir or os.path.join(PLOT_DIR, BASE_NAME + ("" if abs(a.maxdelta - 0.04) < 1e-12 else "-dmgg%g" % a.maxdelta))
    g4 = os.path.join(a.indir, "pi0-tables-g4-%s.npz" % a.suffix)
    cc3 = os.path.join(a.indir, "pi0-tables-cc3-%s.npz" % a.suffix)
    run(g4, cc3, outdir, a.floor, a.maxdelta, a.energy_axis, a.min_count, a.suffix)


if __name__ == "__main__":
    sys.exit(main())
