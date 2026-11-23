#!/usr/bin/env python3
"""
plots_nov5.py -- CC3 vs Geant4 pi0 plots from the NPZ tables written by make_pi0_tables.py.
RUN: python plots_nov5.py            (0.04 files: pi0-tables-{g4,cc3}-full.npz in REC_DIR)

PLOTS (PNG + CSV in the output folder):
  B   truth-side stage ratios (six true-energy bins), CC3/G4
  Bp  reco-side stage ladder (pairs of photon PFOs traced to one generator pi0): CC3/G4 and count / N_true
  1   opening angle, resolved pairs R, full detector      (row 1: alpha_reco counts, row 2: delta-alpha density)
  2   same, fiducial barrel
  3   pre-window reco pair mass, RE, fiducial barrel      (row 1: counts, row 2: density)
  4   mass-window acceptance by region (all / barrel / outside), RE
  5   photon energy response E_reco/E_true, RE, fiducial barrel (higher / lower true-energy photon)

DEFINITIONS (agreed 2026-11-05)
  generator pi0 : PDG 111, two photon daughters, both genStatus 1.
  R  (resolved) : best PFO of each true photon (rank cw*E_pfo), both cw >= floor, two DIFFERENT photon PFOs (PDG 22).
  RE            : R and both PFO energies >= 0.5 GeV.
  window        : |m_unfitted - 0.1349766| <= maxdelta (inclusive).
  true E bins   : 0-5 | 5-10 | 10-20 | 20-30 | 30-50 | 50-125 GeV, lower edge inclusive, upper exclusive. >=125 GeV excluded.
  fiducial barrel: BOTH true photons theta in [40,140] deg and outside the one-sided phi crack windows
                  (centres -157.5+45k deg, dead zone C-0.51 .. C+8.5 deg).  CONFIRM against category_study.py.
  reco-side Bp  : photon PFO whose TOP link (largest cw) is a photon of a generator pi0 with cw >= floor; a pair = one PFO
                  of each of the pi0's two DIFFERENT photons. A pi0 passes a stage if at least one of its pairs passes all
                  stages up to that one. found = that pair is a finder candidate.
Statistics: Poisson / binomial errors, approximate (G4 and CC3 share generator events). No systematics.
"""
import argparse, csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REC_DIR = "/data/dust/user/alimuham/thesis/reco"
PLOT_DIR = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots"
BASE_NAME = "gammagammapi0-nov5"
E_EDGES = [5.0, 10.0, 20.0, 30.0, 50.0, 125.0]
E_NAMES = ["0-5", "5-10", "10-20", "20-30", "30-50", "50-125"]
NB = 6
E_CUT, M_PI0, EPS = 0.5, 0.1349766, 1e-6
G4C, CC3C = "#2166ac", "#d6604d"
THETA_LO, THETA_HI = 40.0, 140.0
CRACK_LO, CRACK_HI = -0.51, 8.5
CRACK_CENTRES = [-157.5 + 45.0 * k for k in range(8)]
NBINS = 30

STAGES = [("R", "two separate photon PFOs"), ("RE", "+ both PFO E >= 0.5 GeV"),
          ("RE_win", "+ pair mass inside window"), ("found", "found by the finder")]

# findings-log / handoff numbers (window 0.04, floor 0.5): per true-E bin 0-5 ... 50-125, [G4, CC3]
LOG = {
 "N_true": {"G4": [15044, 17928, 45662, 38246, 38488, 21038], "CC3": [15044, 17928, 45662, 38244, 38489, 21035]},
 "R":      {"G4": [8558, 10590, 24815, 20114, 19980, 6050],   "CC3": [8546, 10544, 24515, 19965, 19722, 5770]},
 "RE":     {"G4": [5649, 9513, 23315, 19606, 19765, 6016],    "CC3": [5641, 9469, 23066, 19535, 19556, 5746]},
 "RE_win": {"G4": [5218, 8918, 21650, 17617, 15384, 3506],    "CC3": [5172, 8846, 21202, 17272, 14142, 2949]},
 "found":  {"G4": [5202, 8679, 20050, 14391, 10264, 2321],    "CC3": [5154, 8594, 19605, 13557, 9136, 1992]},
}

REQUIRED = ["mc_loop_evt", "mc_pi0_idx", "mc_pi0_E", "mc_true_open_angle", "mc_ph0_gen", "mc_ph1_gen",
            "cand_loop_evt", "cand_d0_pfo_idx", "cand_d1_pfo_idx",
            "pfo_loop_evt", "pfo_pfo_idx", "pfo_E", "pfo_px", "pfo_py", "pfo_pz",
            "pfo_l0_cw", "pfo_l0_pdg", "pfo_l0_idx", "pfo_l0_pi0_par_idx"]
for _k in (0, 1):
    for _c in ("idx", "E", "theta", "phi", "fast"):
        REQUIRED.append("mc_ph%d_%s" % (_k, _c))
    for _c in ("pfo_idx", "cw", "pdg", "E", "px", "py", "pz"):
        REQUIRED.append("mc_ph%d_b0_%s" % (_k, _c))

LINES = []
def say(s=""):
    print(s, flush=True)
    LINES.append(s)

def load(path, tag):
    z = np.load(path)
    a = {k: z[k] for k in z.files if k.startswith(("mc_", "cand_", "pfo_"))}
    miss = [c for c in REQUIRED if c not in a]
    if miss:
        raise KeyError("%s: columns missing in %s: %s" % (tag, path, miss))
    return tag, a

def bin_index(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        b = np.digitize(x, E_EDGES)
    b[~np.isfinite(x)] = -1
    b[b >= NB] = -1
    return b

def pair_key(loop, a, b):
    loop, a, b = (np.asarray(v, dtype=np.int64) for v in (loop, a, b))
    if a.size and max(a.max(), b.max()) >= 2 ** 21:
        raise ValueError("PFO index too large for key packing")
    return (loop << 42) | (np.minimum(a, b) << 21) | np.maximum(a, b)

def cand_keys(a):
    ok = (a["cand_d0_pfo_idx"] >= 0) & (a["cand_d1_pfo_idx"] >= 0)
    return pair_key(a["cand_loop_evt"][ok], a["cand_d0_pfo_idx"][ok], a["cand_d1_pfo_idx"][ok])

def in_crack(phi):
    d = np.zeros(np.shape(phi), dtype=bool)
    for c in CRACK_CENTRES:
        raw = ((phi - c + 180.0) % 360.0) - 180.0
        d |= (raw >= CRACK_LO) & (raw <= CRACK_HI)
    return d

def photon_fiducial(theta, phi):
    th, ph = np.asarray(theta, float), np.asarray(phi, float)
    with np.errstate(invalid="ignore"):
        return (th >= THETA_LO) & (th <= THETA_HI) & ~in_crack(ph) & np.isfinite(th) & np.isfinite(ph)

def counts_by_bin(mask, bins):
    ok = mask & (bins >= 0)
    return np.bincount(bins[ok], minlength=NB)[:NB].astype(np.int64)

def binom_err(k, n):
    return math.sqrt(max(k / n * (1 - k / n), 0.0) / n) if n > 0 else float("nan")

def ratio_err(a, b):
    return a / b * math.sqrt(1.0 / a + 1.0 / b) if a > 0 and b > 0 else float("nan")

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ------------------------------------------------------------------ per-file state (truth-side)
def build(tag, a, floor, maxdelta):
    gen = (a["mc_ph0_gen"] == 1) & (a["mc_ph1_gen"] == 1)
    ia, ib = a["mc_ph0_b0_pfo_idx"].astype(np.int64), a["mc_ph1_b0_pfo_idx"].astype(np.int64)
    ma = (ia >= 0) & (a["mc_ph0_b0_cw"].astype(float) >= floor - EPS)
    mb = (ib >= 0) & (a["mc_ph1_b0_cw"].astype(float) >= floor - EPS)
    both = ma & mb
    merged = both & (ia == ib)
    other = both & ~merged & ((a["mc_ph0_b0_pdg"] != 22) | (a["mc_ph1_b0_pdg"] != 22))
    R = both & ~merged & ~other
    v0 = np.stack([a["mc_ph0_b0_px"], a["mc_ph0_b0_py"], a["mc_ph0_b0_pz"]], axis=1).astype(float)
    v1 = np.stack([a["mc_ph1_b0_px"], a["mc_ph1_b0_py"], a["mc_ph1_b0_pz"]], axis=1).astype(float)
    e0, e1 = a["mc_ph0_b0_E"].astype(float), a["mc_ph1_b0_E"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m2 = (e0 + e1) ** 2 - np.sum((v0 + v1) ** 2, axis=1)
        mass = np.sqrt(np.clip(m2, 0.0, None))
        ang = np.arctan2(np.linalg.norm(np.cross(v0, v1), axis=1), np.sum(v0 * v1, axis=1))
    RE = R & (e0 >= E_CUT) & (e1 >= E_CUT)
    win = RE & (mass >= M_PI0 - maxdelta) & (mass <= M_PI0 + maxdelta)
    found = np.zeros(R.size, dtype=bool)
    idx = np.where(R)[0]
    if idx.size:
        found[idx] = np.isin(pair_key(a["mc_loop_evt"][idx], ia[idx], ib[idx]), cand_keys(a))
    fid = photon_fiducial(a["mc_ph0_theta"], a["mc_ph0_phi"]) & photon_fiducial(a["mc_ph1_theta"], a["mc_ph1_phi"])
    f0, f1 = a["mc_ph0_fast"] == 1, a["mc_ph1_fast"] == 1
    S = dict(tag=tag, a=a, gen=gen, R=R, RE=RE, win=win, found=found, mass=mass, ang=ang,
             tru=a["mc_true_open_angle"].astype(float), e0=e0, e1=e1, fid=fid,
             fast_any=f0 | f1, fast_both=f0 & f1, bin=bin_index(a["mc_pi0_E"]))
    c = {"N_true": counts_by_bin(gen, S["bin"]), "R": counts_by_bin(gen & R, S["bin"]),
         "RE": counts_by_bin(gen & RE, S["bin"]), "RE_win": counts_by_bin(gen & win, S["bin"]),
         "found": counts_by_bin(gen & found, S["bin"])}
    S["counts"] = c
    S["n_over125"] = int((gen & (np.asarray(a["mc_pi0_E"], float) >= E_EDGES[-1])).sum())
    S["found_outside_window"] = int((gen & found & ~win).sum())
    S["found_fail_E"] = int((gen & found & ~RE).sum())
    return S

# ------------------------------------------------------------------ reco-side ladder (Bp)
def reco_side(S, floor, maxdelta):
    a = S["a"]
    n = a["mc_loop_evt"].size
    if a["mc_pi0_idx"].size and a["mc_pi0_idx"].max() >= 2 ** 24:
        raise ValueError("pi0 index too large for key packing")
    mk = (a["mc_loop_evt"].astype(np.int64) << 24) | a["mc_pi0_idx"].astype(np.int64)
    order = np.argsort(mk, kind="stable")
    mks = mk[order]
    cw = a["pfo_l0_cw"].astype(float)
    par = a["pfo_l0_pi0_par_idx"].astype(np.int64)
    ok = (cw >= floor - EPS) & (a["pfo_l0_pdg"] == 22) & (par >= 0)
    pk = (a["pfo_loop_evt"].astype(np.int64) << 24) | np.where(par >= 0, par, 0)
    pos = np.clip(np.searchsorted(mks, pk), 0, max(n - 1, 0))
    hit = ok & (mks[pos] == pk) & S["gen"][order[pos]]
    row = order[pos]
    l0 = a["pfo_l0_idx"]
    s0 = hit & (a["mc_ph0_idx"][row] == l0)
    s1 = hit & (a["mc_ph1_idx"][row] == l0)
    i0, i1 = np.where(s0)[0], np.where(s1)[0]
    k0, k1 = pk[i0], pk[i1]
    o1 = np.argsort(k1, kind="stable")
    k1s = k1[o1]
    lo, hi = np.searchsorted(k1s, k0, "left"), np.searchsorted(k1s, k0, "right")
    cnt = hi - lo
    tot = int(cnt.sum())
    rep = np.repeat(np.arange(i0.size), cnt)
    off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    p0 = i0[rep]
    p1 = i1[o1[lo[rep] + off]]
    r = row[p0]
    E0, E1 = a["pfo_E"][p0].astype(float), a["pfo_E"][p1].astype(float)
    px = a["pfo_px"][p0].astype(float) + a["pfo_px"][p1]
    py = a["pfo_py"][p0].astype(float) + a["pfo_py"][p1]
    pz = a["pfo_pz"][p0].astype(float) + a["pfo_pz"][p1]
    with np.errstate(invalid="ignore"):
        mass = np.sqrt(np.clip((E0 + E1) ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0, None))
    s_re = (E0 >= E_CUT) & (E1 >= E_CUT)
    s_win = s_re & (mass >= M_PI0 - maxdelta) & (mass <= M_PI0 + maxdelta)
    incand = np.isin(pair_key(a["pfo_loop_evt"][p0], a["pfo_pfo_idx"][p0], a["pfo_pfo_idx"][p1]), cand_keys(a))
    per = {}
    for name, f in (("R", np.ones(r.size, bool)), ("RE", s_re), ("RE_win", s_win), ("found", incand)):
        m = np.zeros(n, dtype=bool)
        m[r[f]] = True
        per[name] = m
    c = {"N_true": S["counts"]["N_true"]}
    for name in per:
        c[name] = counts_by_bin(S["gen"] & per[name], S["bin"])
    extra = dict(n_pairs=int(tot), truth_R_not_in_reco_R=int((S["gen"] & S["R"] & ~per["R"]).sum()),
                 reco_R_not_in_truth_R=int((S["gen"] & per["R"] & ~S["R"]).sum()),
                 cand_outside_window=int((incand & ~s_win).sum()))
    return c, extra

# ------------------------------------------------------------------ fast-sim helper
def fast_text(S, m):
    if S["tag"] != "CC3":
        return None
    n = int(m.sum())
    if n == 0:
        return "CC3: no pairs"
    return "CC3: %.1f%% of pairs have >=1 fast-sim photon (both: %.1f%%)" % (
        100.0 * S["fast_any"][m].mean(), 100.0 * S["fast_both"][m].mean())

# ------------------------------------------------------------------ B and Bp
def stage_xticks(ax):
    ax.set_xticks(range(NB))
    ax.set_xticklabels(E_NAMES)
    ax.set_xlabel("true pi0 energy [GeV]")

def ratio_panel(ax, cG, cC, keys_labels, title):
    allv = []
    series = [("N_true", "all generator pi0s", "0.55", "x", -0.18)] + \
             [(k, l, c, m, o) for (k, l), c, m, o in zip(keys_labels, ("#1b9e77", "#7570b3", "#e6ab02", "black"),
                                                      ("o", "s", "^", "D"), (-0.06, 0.02, 0.10, 0.18))]
    for key, lab, col, mk, off in series:
        r = np.full(NB, np.nan); e = np.full(NB, np.nan)
        for b in range(NB):
            g, c = cG[key][b], cC[key][b]
            if g > 0 and c > 0:
                r[b], e[b] = c / g, ratio_err(c, g)
        ax.errorbar(np.arange(NB) + off, r, yerr=e, fmt=mk + "-", color=col, ms=5, lw=1, label=lab)
        allv += list(r[np.isfinite(r)] - e[np.isfinite(e)]) + list(r[np.isfinite(r)] + e[np.isfinite(e)])
    ax.axhline(1.0, color="gray", lw=1)
    ax.set_ylim(min(0.75, min(allv) - 0.02) if allv else 0.75, max(1.08, max(allv) + 0.02) if allv else 1.08)
    ax.set_ylabel("CC3 / G4 (pi0s still in the chain)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title(title, fontsize=9)
    stage_xticks(ax); clean(ax)

def plot_B(SG, SC, outdir, note):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ratio_panel(ax, SG["counts"], SC["counts"], STAGES, "Plot B (truth side): where does CC3 fall behind G4?")
    fig.text(0.01, 0.01, note, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(outdir, "B_stage_ratios_truth_side.png"), dpi=150)
    plt.close(fig)

def plot_Bp(cG, cC, outdir, note):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ratio_panel(ax, cG, cC, STAGES, "Plot B' (reco side): CC3 / G4")
    for key, lab, mk in (("R", "R'", "o"), ("RE", "RE'", "s"), ("RE_win", "RE' + window", "^"), ("found", "found", "D")):
        for c, col, ls in ((cG, G4C, "-"), (cC, CC3C, "--")):
            f = np.array([c[key][b] / c["N_true"][b] if c["N_true"][b] > 0 else np.nan for b in range(NB)])
            ax2.plot(np.arange(NB), f, mk + ls, color=col, ms=4, lw=1,
                     label=("G4 " if col == G4C else "CC3 ") + lab)
    ax2.set_ylabel("pi0s at stage / generator pi0s (common denominator)")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Plot B' (reco side): fraction of generator pi0s", fontsize=9)
    ax2.legend(frameon=False, fontsize=6, ncol=2)
    stage_xticks(ax2); clean(ax2)
    fig.text(0.01, 0.01, note, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(outdir, "Bp_stage_ladder_reco_side.png"), dpi=150)
    plt.close(fig)

# ------------------------------------------------------------------ generic grid of distributions
def grid_fig(SG, SC, rows, sel, fname, title, outdir, cache, note, fast_row=None, fast_rows_csv=None, tag=""):
    fig, axes = plt.subplots(len(rows), NB, figsize=(3.7 * NB, 3.4 * len(rows) + 1.2), squeeze=False)
    for b in range(NB):
        mg, mc = sel(SG, b), sel(SC, b)
        for ri, row in enumerate(rows):
            ax = axes[ri, b]
            ag, ac = row["get"](SG, mg), row["get"](SC, mc)
            ag, ac = ag[np.isfinite(ag)], ac[np.isfinite(ac)]
            if ri == 0:
                ax.set_title("true pi0 E %s GeV" % E_NAMES[b], fontsize=9)
            if ag.size == 0 and ac.size == 0:
                ax.text(0.5, 0.5, "no pairs", transform=ax.transAxes, ha="center"); clean(ax); continue
            ck = (row["key"], b)
            if ck in cache:
                edges = cache[ck]
            elif row.get("edges") is not None:
                edges = np.asarray(row["edges"], float)
            else:
                pool = np.concatenate([x for x in (ag, ac) if x.size])
                lo, hi = np.percentile(pool, [0.5, 99.5])
                if hi <= lo:
                    hi = lo + 1.0
                edges = np.linspace(lo, hi, NBINS + 1)
            cache[ck] = edges
            w = np.diff(edges)
            for x, col, ls, lab in ((ag, G4C, "-", "G4"), (ac, CC3C, "--", "CC3")):
                h, _ = np.histogram(x, bins=edges)
                under, over = int((x < edges[0]).sum()), int((x > edges[-1]).sum())
                y = h.astype(float) if row["kind"] == "counts" else h / (max(x.size, 1) * w)
                ax.step(edges, np.append(y, y[-1]), where="post", color=col, ls=ls, lw=1.6,
                        label="%s N=%d%s" % (lab, x.size, (" (u%d/o%d)" % (under, over)) if (under or over) else ""))
            for v in row.get("vlines", []):
                ax.axvline(v, color="k", ls=":", lw=1)
            ax.set_xlabel(row["xlabel"], fontsize=8)
            ax.set_ylabel(row["ylabel"] if b == 0 else "", fontsize=8)
            ax.legend(frameon=False, fontsize=7)
            if ri == 0 and row.get("text_true"):
                tg = SG["tru"][mg] * 1e3; tc = SC["tru"][mc] * 1e3
                ax.text(0.02, 0.62, "median alpha_true [mrad]\nG4 %.2f | CC3 %.2f" % (
                    np.median(tg) if tg.size else np.nan, np.median(tc) if tc.size else np.nan),
                    transform=ax.transAxes, fontsize=7, va="top")
            if ri == 0 and fast_row:
                t = fast_text(SC, mc)
                if t:
                    ax.text(0.02, 0.40, t.replace(" (both", "\n(both"), transform=ax.transAxes, fontsize=7, va="top")
                    fast_rows_csv.append([tag, E_NAMES[b], int(mc.sum()),
                                          float(SC["fast_any"][mc].mean()) if mc.any() else float("nan"),
                                          float(SC["fast_both"][mc].mean()) if mc.any() else float("nan")])
            clean(ax)
    fig.suptitle(title, fontsize=11)
    fig.text(0.01, 0.005, note, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(outdir, fname), dpi=120)
    plt.close(fig)

def plot_5(SG, SC, outdir, note, fastcsv):
    sel = lambda S, b: S["gen"] & S["RE"] & S["fid"] & (S["bin"] == b)
    ed = np.linspace(0.4, 1.8, 57)
    def mk(k, lab):
        return dict(key="E%d" % k, xlabel=lab, ylabel="density", kind="density", edges=ed,
                    get=lambda S, m, k=k: (S["e%d" % k] / S["a"]["mc_ph%d_E" % k])[m])
    rows = [mk(0, "E_reco/E_true, higher-true-E photon"), mk(1, "E_reco/E_true, lower-true-E photon")]
    grid_fig(SG, SC, rows, sel, "5_energy_response_barrel.png",
             "Plot 5: photon energy response, RE, fiducial barrel", outdir, {}, note, fast_row=True,
             fast_rows_csv=fastcsv, tag="plot5")

# ------------------------------------------------------------------ plot 4
def plot_4(SG, SC, maxdelta, outdir, note, fastcsv):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
    rows_csv = []
    for b in range(NB):
        ax = axes[b // 3, b % 3]
        regs = [("all detector", lambda S: S["gen"] & S["RE"] & (S["bin"] == b)),
                ("fiducial barrel", lambda S: S["gen"] & S["RE"] & S["fid"] & (S["bin"] == b)),
                ("outside barrel", lambda S: S["gen"] & S["RE"] & ~S["fid"] & (S["bin"] == b))]
        for j, S in enumerate((SG, SC)):
            for i, (nm, f) in enumerate(regs):
                m = f(S)
                n = int(m.sum()); k = int((m & S["win"]).sum())
                p = 100.0 * k / n if n else np.nan
                e = 100.0 * binom_err(k, n) if n else np.nan
                ax.bar(i + (-0.2 if j == 0 else 0.2), p, width=0.38, color=G4C if j == 0 else CC3C,
                       yerr=e, capsize=2, label=S["tag"] if i == 0 else None)
                ax.text(i + (-0.2 if j == 0 else 0.2), (p if np.isfinite(p) else 0) + 1, "N=%d" % n, rotation=90,
                        ha="center", va="bottom", fontsize=6)
                rows_csv.append([E_NAMES[b], S["tag"], nm, n, k, p])
        t = fast_text(SC, regs[1][1](SC))
        ax.set_title("true pi0 E %s GeV" % E_NAMES[b] + ("\nbarrel " + t if t else ""), fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in regs], fontsize=8)
        ax.set_ylim(0, 125); ax.set_ylabel("RE pairs inside +-%.0f MeV [%%]" % (1000 * maxdelta), fontsize=8)
        ax.legend(frameon=False, fontsize=8, loc="upper right"); clean(ax)
    fig.suptitle("Plot 4: mass-window acceptance by region (RE pairs)", fontsize=11)
    fig.text(0.01, 0.005, note, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(outdir, "4_window_acceptance_by_region.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(outdir, "window_acceptance.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["E_bin", "file", "region", "N_RE", "N_in_window", "pass_percent"]); w.writerows(rows_csv)

# ------------------------------------------------------------------ main
def check_log(SG, SC):
    say("\n[closure vs findings log / handoff, window 0.04, floor 0.5]")
    bad = 0
    for key in ("N_true", "R", "RE", "RE_win", "found"):
        for S in (SG, SC):
            got = [int(x) for x in S["counts"][key]]
            exp = LOG[key][S["tag"]]
            ok = got == exp
            bad += (not ok)
            say("  %-7s %-3s %s %s" % (key, S["tag"], "MATCH" if ok else "DIFFERENT", "" if ok else "got %s log %s" % (got, exp)))
    say("  totals N_true: G4 %d CC3 %d (log 176406 / 176402)" % (SG["counts"]["N_true"].sum() + SG["n_over125"], SC["counts"]["N_true"].sum() + SC["n_over125"]))
    say("  => %s" % ("ALL MATCH" if not bad else "%d DIFFERENCES: read them before using the plots" % bad))

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    g4p = os.path.join(args.indir, "pi0-tables-g4-%s.npz" % args.suffix)
    c3p = os.path.join(args.indir, "pi0-tables-cc3-%s.npz" % args.suffix)
    say("plots_nov5.py | floor %.2f | window %.4f-%.4f GeV | suffix %s" % (args.floor, M_PI0 - args.maxdelta, M_PI0 + args.maxdelta, args.suffix))
    say("G4 : " + g4p); say("CC3: " + c3p)
    SG = build(*load(g4p, "G4"), args.floor, args.maxdelta)
    SC = build(*load(c3p, "CC3"), args.floor, args.maxdelta)
    for S in (SG, SC):
        say("%s: generator pi0s %d (>=125 GeV excluded: %d) | found pairs outside window: %d | found failing E cut: %d" % (
            S["tag"], int(S["gen"].sum()), S["n_over125"], S["found_outside_window"], S["found_fail_E"]))
    if args.suffix == "full" and abs(args.maxdelta - 0.04) < 1e-12 and abs(args.floor - 0.5) < 1e-12:
        check_log(SG, SC)
    else:
        say("(closure vs log skipped: not the 0.04 / floor 0.5 / full setting)")

    rows = []
    for S in (SG, SC):
        for key in ("N_true", "R", "RE", "RE_win", "found"):
            rows.append([S["tag"], key] + [int(x) for x in S["counts"][key]])
    with open(os.path.join(args.outdir, "stage_counts_truth_side.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "stage"] + E_NAMES); w.writerows(rows)

    note_def = ("Generator pi0s (both photons genStatus 1). R: best PFO per true photon (rank cw*E_pfo), cw>=%.1f, two different photon PFOs. "
                "RE: both PFO E>=0.5 GeV. Window: %.1f +- %.0f MeV on the unfitted PFO-pair mass. Errors approximate (shared events)." % (
                    args.floor, 1000 * M_PI0, 1000 * args.maxdelta))
    plot_B(SG, SC, args.outdir, note_def)

    cG, eG = reco_side(SG, args.floor, args.maxdelta)
    cC, eC = reco_side(SC, args.floor, args.maxdelta)
    say("\n[reco-side ladder B']")
    for t, e in (("G4", eG), ("CC3", eC)):
        say("  %s: pairs %d | truth-side R pi0s missing on reco side: %d (log: 0) | reco-side-only R pi0s: %d (log: 1407 G4 / 1510 CC3 at floor 0.5) | candidate pairs outside window: %d" % (
            t, e["n_pairs"], e["truth_R_not_in_reco_R"], e["reco_R_not_in_truth_R"], e["cand_outside_window"]))
    rows = []
    for t, c in (("G4", cG), ("CC3", cC)):
        for key in ("N_true", "R", "RE", "RE_win", "found"):
            rows.append([t, key] + [int(x) for x in c[key]])
    with open(os.path.join(args.outdir, "stage_counts_reco_side.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "stage"] + E_NAMES); w.writerows(rows)
    fo_t = (SC["counts"]["found"].sum() / SG["counts"]["found"].sum())
    fo_r = (cC["found"].sum() / cG["found"].sum())
    say("  found CC3/G4 all bins <125 GeV: truth side %.3f | reco side %.3f ; R: truth %.3f | reco %.3f" % (
        fo_t, fo_r, SC["counts"]["R"].sum() / SG["counts"]["R"].sum(), cC["R"].sum() / cG["R"].sum()))
    plot_Bp(cG, cC, args.outdir, note_def.replace("R: best PFO per true photon (rank cw*E_pfo), cw>=%.1f, two different photon PFOs." % args.floor,
            "R': top-linked photon PFOs of the two different photons of one generator pi0 (cw>=%.1f); a pi0 passes if at least one pair passes." % args.floor))

    cache = {}
    fastcsv = []
    ang_rows = lambda: [
        dict(key="ar", xlabel="alpha_reco [mrad]", ylabel="pairs per bin (raw)", kind="counts", text_true=True,
             get=lambda S, m: S["ang"][m] * 1e3),
        dict(key="da", xlabel="alpha_reco - alpha_true [mrad]", ylabel="density (unit area, all entries)", kind="density",
             get=lambda S, m: (S["ang"][m] - S["tru"][m]) * 1e3)]
    sel1 = lambda S, b: S["gen"] & S["R"] & (S["bin"] == b) & np.isfinite(S["ang"]) & np.isfinite(S["tru"])
    sel2 = lambda S, b: sel1(S, b) & S["fid"]
    grid_fig(SG, SC, ang_rows(), sel1, "1_opening_angle_full_detector.png",
             "Plot 1: opening angle, resolved pairs R, full detector", args.outdir, cache, note_def)
    grid_fig(SG, SC, ang_rows(), sel2, "2_opening_angle_fiducial_barrel.png",
             "Plot 2: opening angle, resolved pairs R, fiducial barrel (fast-sim-enriched, not pure fast-sim)", args.outdir, cache,
             note_def + " Fiducial barrel: both true photons theta 40-140 deg, outside phi cracks. Histogram edges identical to Plot 1.",
             fast_row=True, fast_rows_csv=fastcsv, tag="plot2")
    sel3 = lambda S, b: S["gen"] & S["RE"] & S["fid"] & (S["bin"] == b)
    me = np.linspace(0.0, 300.0, 61)
    mrows = [dict(key="mc", xlabel="m_gg reco [MeV]", ylabel="pairs per 5 MeV (raw)", kind="counts", edges=me,
                  vlines=[1000 * (M_PI0 - args.maxdelta), 1000 * (M_PI0 + args.maxdelta)], get=lambda S, m: S["mass"][m] * 1e3),
             dict(key="md", xlabel="m_gg reco [MeV]", ylabel="density (unit area, all entries)", kind="density", edges=me,
                  vlines=[1000 * (M_PI0 - args.maxdelta), 1000 * (M_PI0 + args.maxdelta)], get=lambda S, m: S["mass"][m] * 1e3)]
    grid_fig(SG, SC, mrows, sel3, "3_pair_mass_before_window_barrel.png",
             "Plot 3: reco pair mass before any window or finder cut, RE, fiducial barrel", args.outdir, {},
             note_def + " u/o = entries below 0 / above 300 MeV (counted in N and in the density normalisation).",
             fast_row=True, fast_rows_csv=fastcsv, tag="plot3")
    plot_4(SG, SC, args.maxdelta, args.outdir, note_def, fastcsv)
    plot_5(SG, SC, args.outdir, note_def + " u/o = outside 0.4-1.8.", fastcsv)
    with open(os.path.join(args.outdir, "fastsim_fraction_cc3.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["plot", "E_bin", "N_pairs", "frac_ge1_fastsim_photon", "frac_both_fastsim"]); w.writerows(fastcsv)
    say("\nfiles written to " + args.outdir)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")

def main(argv=None):
    ap = argparse.ArgumentParser(description="Run with no arguments for the 0.04 full files.")
    ap.add_argument("--indir", default=REC_DIR)
    ap.add_argument("--suffix", default="full")
    ap.add_argument("--maxdelta", type=float, default=0.04)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--outdir", default=os.path.join(PLOT_DIR, BASE_NAME))
    run(ap.parse_args(argv))

if __name__ == "__main__":
    sys.exit(main())
