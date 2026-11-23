#!/usr/bin/env python3
"""
endcap_plots_nov9.py (v2, 9 Nov 2026) -- five presentation plots, G4 vs CC3, barrel vs endcap. Reads the NPZ tables of make_pi0_tables.py (no ROOT).

RUN:  python3 endcap_plots_nov9.py                     (tables pi0-tables-{g4,cc3}-full.npz, finder window +-0.04 GeV)
      python3 endcap_plots_nov9.py --suffix first1000   (quick test on the 1k tables)
options: --indir DIR --suffix S --maxdelta GeV --floor 0.5 --endcap {both_photons,pi0_dir} --outdir DIR
         --min-n 30 --angle-bin 0.25 --theta-bin 1.0 --resp-ylim LO HI --ratio-ylim LO HI --dpi 150

CHANGES vs v1: plot titles state what is plotted (no questions); plot 2 uses ONE common x range and bin width (--angle-bin, mrad) and ONE shared
y axis in all six panels; plot 4 shares y within a column; plot 3 has fixed shared y axes (--resp-ylim, --ratio-ylim) and warns in summary.txt
if a point falls outside; cells with fewer than --min-n entries now print a warning instead of being skipped silently.

SAMPLE (plots 1-4): generator pi0s (both photons genStatus 1); two separate photon PFOs (best PFO per photon, rank cw*E_pfo, cw >= floor,
both PDG 22, different PFOs); both PFO energies >= 0.5 GeV; NO mass window (the window is only the numerator of plot 1).
REGIONS:  barrel = BOTH photons theta 40-140 deg and outside the one-sided phi dead windows [C-0.51, C+8.5], C = -157.5+45k (as category_study.py).
          endcap (default, option A "both_photons") = BOTH photons in folded theta [15,30) deg, both on the same side, all phi.
          endcap option B "pi0_dir" = pi0 direction in folded theta [15,30) deg, both photons on the same side.
ENERGY:   true pi0 energy, bins 5-10, 10-20, 20-30, 30-50, 50-125 GeV (lower edge inclusive).
FAST-SIM: CC3 file only. full = both true photons flagged, partial = EXACTLY one flagged.

OUTPUT:   plot1_acceptance_ratio.png, plot2_opening_angle.png, plot3_energy_response.png, plot4_mass.png, plot5_theta.png,
          one CSV per plot (the numbers behind it), notes.md (plain-language notes per plot, NOT for the slides), summary.txt (self-checks, warnings).
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REC_DIR = "/data/dust/user/alimuham/thesis/reco"
PLOT_DIR = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots"
E_EDGES = [5.0, 10.0, 20.0, 30.0, 50.0, 125.0]
E_NAMES = ["5-10", "10-20", "20-30", "30-50", "50-125"]
GRID_BINS = [2, 3, 4]
E_CUT = 0.5
M_PI0 = 0.1349766
EPS = 1e-6
E_PHOTON_MIN = 10.0
THETA_BARREL = (40.0, 140.0)
CRACK_CENTERS = [-157.5 + 45.0 * k for k in range(8)]
CRACK_LO, CRACK_HI = -0.51, 8.5
ENDCAP_FOLDED = (15.0, 30.0)
G4_COL, CC3_COL = "#2166ac", "#d6604d"
REG_STYLE = {"barrel": ("#4d4d4d", "o"), "endcap": ("#1b7837", "s")}
REGIONS = ["barrel", "endcap"]
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


def finite(v):
    v = np.asarray(v, dtype=float)
    return v[np.isfinite(v)]


def centres():
    return [math.sqrt(E_EDGES[i] * E_EDGES[i + 1]) for i in range(len(E_NAMES))]


def nice_range(vals, step, qlo=0.5, qhi=99.5):
    vals = finite(vals)
    if vals.size < 20:
        return None
    lo, hi = np.percentile(vals, [qlo, qhi])
    lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
    return (lo, lo + step) if hi <= lo else (lo, hi)


# ------------------------------------------------------------------ geometry
def phi_in_dead_window(phi_deg):
    phi = np.asarray(phi_deg, dtype=float)
    dead = np.zeros(phi.shape, dtype=bool)
    for c in CRACK_CENTERS:
        raw = ((phi - c + 180.0) % 360.0) - 180.0
        dead |= (raw >= CRACK_LO) & (raw <= CRACK_HI)
    return dead


def photon_in_barrel(theta_deg, phi_deg):
    th, ph = np.asarray(theta_deg, dtype=float), np.asarray(phi_deg, dtype=float)
    ok = np.isfinite(th) & np.isfinite(ph)
    return ok & (th >= THETA_BARREL[0]) & (th <= THETA_BARREL[1]) & ~phi_in_dead_window(ph)


def fold(th):
    th = np.asarray(th, dtype=float)
    return np.minimum(th, 180.0 - th)


def in_endcap_band(th):
    f = fold(th)
    return np.isfinite(f) & (f >= ENDCAP_FOLDED[0]) & (f < ENDCAP_FOLDED[1])


def same_side(t0, t1):
    t0, t1 = np.asarray(t0, dtype=float), np.asarray(t1, dtype=float)
    return ((t0 < 90.0) & (t1 < 90.0)) | ((t0 > 90.0) & (t1 > 90.0))


# ------------------------------------------------------------------ loading and state
def pair_key(loop, a, b):
    loop, a, b = (np.asarray(v, dtype=np.int64) for v in (loop, a, b))
    if a.size and (max(a.max(), b.max()) >= 2 ** 21):
        raise ValueError("PFO index too large for the key packing")
    return (loop << 42) | (np.minimum(a, b) << 21) | np.maximum(a, b)


def load_tables(path, need_pi0_theta):
    z = np.load(path)
    a = {k: z[k] for k in z.files if k.startswith(("mc_", "cand_"))}
    req = REQUIRED + (["mc_pi0_theta"] if need_pi0_theta else [])
    missing = [c for c in req if c not in a]
    if missing:
        raise KeyError("columns missing in %s: %s" % (path, missing))
    return a


def build_state(a, tag, floor, maxdelta, endcap_mode):
    n = len(a["mc_loop_evt"])
    ia, ib = a["mc_ph0_b0_pfo_idx"].astype(np.int64), a["mc_ph1_b0_pfo_idx"].astype(np.int64)
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
        m2 = (e0 + e1) ** 2 - np.sum((v0 + v1) ** 2, axis=1)
        mass = np.sqrt(np.clip(m2, 0.0, None))
        th_reco = np.arctan2(np.linalg.norm(np.cross(v0, v1), axis=1), np.sum(v0 * v1, axis=1)) * 1000.0
        r0 = e0 / a["mc_ph0_E"].astype(float)
        r1 = e1 / a["mc_ph1_E"].astype(float)
    th_true = a["mc_true_open_angle"].astype(float) * 1000.0
    res_E = res & (e0 >= E_CUT) & (e1 >= E_CUT)
    res_E_window = res_E & (mass >= M_PI0 - maxdelta) & (mass <= M_PI0 + maxdelta)
    found = np.zeros(n, dtype=bool)
    idx = np.where(res)[0]
    if idx.size:
        ok = (a["cand_d0_pfo_idx"] >= 0) & (a["cand_d1_pfo_idx"] >= 0)
        ck = pair_key(a["cand_loop_evt"][ok], a["cand_d0_pfo_idx"][ok], a["cand_d1_pfo_idx"][ok])
        found[idx] = np.isin(pair_key(a["mc_loop_evt"][idx], ia[idx], ib[idx]), ck)
    gen = (a["mc_ph0_gen"] == 1) & (a["mc_ph1_gen"] == 1)
    t0, t1 = a["mc_ph0_theta"].astype(float), a["mc_ph1_theta"].astype(float)
    barrel = gen & photon_in_barrel(t0, a["mc_ph0_phi"]) & photon_in_barrel(t1, a["mc_ph1_phi"])
    if endcap_mode == "both_photons":
        endcap = gen & same_side(t0, t1) & in_endcap_band(t0) & in_endcap_band(t1)
    else:
        endcap = gen & same_side(t0, t1) & in_endcap_band(a["mc_pi0_theta"])
    d = np.digitize(a["mc_pi0_E"].astype(float), E_EDGES)
    return dict(tag=tag, a=a, n=n, res=res, res_E=res_E, res_E_window=res_E_window, found=found, mass=mass * 1000.0,
                dtheta=th_reco - th_true, r0=r0, r1=r1, gen=gen, region={"barrel": barrel, "endcap": endcap},
                bin=np.where((d >= 1) & (d <= len(E_NAMES)), d - 1, -1),
                f0=(a["mc_ph0_fast"] == 1), f1=(a["mc_ph1_fast"] == 1),
                n_flag=(a["mc_ph0_fast"] == 1).astype(int) + (a["mc_ph1_fast"] == 1).astype(int))


def cell(S, reg, b):
    return S["region"][reg] & S["res_E"] & (S["bin"] == b)


def pair_shares(S, m):
    n = int(m.sum())
    if n == 0:
        return float("nan"), float("nan")
    return 100.0 * float((S["n_flag"][m] == 2).sum()) / n, 100.0 * float((S["n_flag"][m] == 1).sum()) / n


def photon_share(S, m):
    n = int(m.sum())
    if n == 0:
        return float("nan")
    return 100.0 * float(S["f0"][m].sum() + S["f1"][m].sum()) / (2 * n)


def pct(x):
    return "-" if not np.isfinite(x) else "%d" % int(round(x))


def strip(ax, xs, texts, y, color, label):
    tr = ax.get_xaxis_transform()
    for x, t in zip(xs, texts):
        ax.text(x, y, t, transform=tr, ha="center", va="top", fontsize=8, color=color)
    if label:
        ax.text(-0.02, y, label, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=color)


def set_energy_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(E_EDGES[0], E_EDGES[-1])
    ax.set_xticks(E_EDGES)
    ax.set_xticklabels(["%g" % e for e in E_EDGES])
    ax.minorticks_off()


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ------------------------------------------------------------------ plot 1
def plot1(Sg, Sc, args, outdir):
    xc = centres()
    fig, ax = plt.subplots(figsize=(9, 6.2))
    rows, allv = [], []
    for reg, dodge in zip(REGIONS, (0.97, 1.03)):
        col, mk = REG_STYLE[reg]
        xs, ys, es, txt = [], [], [], []
        for b in range(len(E_NAMES)):
            mg, mc = cell(Sg, reg, b), cell(Sc, reg, b)
            ng, nc = int(mg.sum()), int(mc.sum())
            kg, kc = int((mg & Sg["res_E_window"]).sum()), int((mc & Sc["res_E_window"]).sum())
            full, part = pair_shares(Sc, mc)
            txt.append("%s | %s" % (pct(full), pct(part)))
            ratio = err = float("nan")
            if min(ng, nc) >= args.min_n and kg > 0 and kc > 0:
                pg, pc = kg / ng, kc / nc
                ratio = pc / pg
                err = ratio * math.sqrt((1 - pg) / kg + (1 - pc) / kc)
                xs.append(xc[b] * dodge); ys.append(ratio); es.append(err)
                allv += [ratio - err, ratio + err]
            else:
                print("WARNING plot1 %s %s GeV: not drawn (pairs G4/CC3 %d/%d, inside window G4/CC3 %d/%d, min-n %d)" % (reg, E_NAMES[b], ng, nc, kg, kc, args.min_n))
            rows.append([reg, E_NAMES[b], ng, kg, nc, kc, (kg / ng if ng else float("nan")), (kc / nc if nc else float("nan")),
                         ratio, err, full, part])
        ax.errorbar(xs, ys, yerr=es, fmt="-" + mk, color=col, ms=6, lw=1.4, capsize=3, label=reg)
        strip(ax, xc, txt, -0.19 if reg == "barrel" else -0.26, col, "CC3 fast-sim, %s (full | partial %%)" % reg)
    ax.axhline(1.0, color="gray", lw=1)
    set_energy_axis(ax)
    lo, hi = (min(allv + [1.0]), max(allv + [1.0])) if allv else (0.5, 1.1)
    ax.set_ylim(lo - 0.05, hi + 0.05)
    ax.set_xlabel("true $\\pi^0$ energy [GeV]")
    ax.set_ylabel("CC3 / G4  (pairs inside the mass window / all resolved pairs)")
    ax.set_title("Mass-window acceptance ratio CC3 / G4 vs true $\\pi^0$ energy\n"
                 "(resolved pairs with reco mass in 134.98 \u00b1 %d MeV, divided by all resolved pairs)" % round(1000 * args.maxdelta), fontsize=11)
    ax.legend(frameon=False, loc="lower left")
    clean(ax)
    fig.subplots_adjust(left=0.33, right=0.97, bottom=0.30, top=0.90)
    fig.savefig(os.path.join(outdir, "plot1_acceptance_ratio.png"), dpi=args.dpi)
    plt.close(fig)
    write_csv(os.path.join(outdir, "plot1_acceptance_ratio.csv"),
              ["region", "true_E_GeV", "G4_pairs", "G4_inside_window", "CC3_pairs", "CC3_inside_window", "G4_acceptance",
               "CC3_acceptance", "ratio_CC3_over_G4", "ratio_error", "CC3_full_fastsim_pct", "CC3_partial_fastsim_pct"], rows)


# ------------------------------------------------------------------ plots 2 and 4 (grids)
def grid_plot(kind, Sg, Sc, args, outdir):
    fig, axes = plt.subplots(2, len(GRID_BINS), figsize=(15, 8.2))
    rows, tops = [], {}
    common = None
    if kind == "angle":
        pooled = np.concatenate([finite(S["dtheta"][cell(S, reg, b)]) for b in GRID_BINS for S in (Sg, Sc) for reg in REGIONS])
        rng = nice_range(pooled, args.angle_bin)
        if rng is not None:
            common = np.arange(rng[0], rng[1] + 1e-9, args.angle_bin)
    for ci, b in enumerate(GRID_BINS):
        edges = common if kind == "angle" else np.arange(20.0, 265.0, 5.0)
        for ri, reg in enumerate(REGIONS):
            ax = axes[ri][ci]
            full, part = pair_shares(Sc, cell(Sc, reg, b))
            ax.set_title("%s, %s GeV\nCC3 fast-sim: full %s%% | partial %s%%" % (reg, E_NAMES[b], pct(full), pct(part)), fontsize=10)
            if edges is None:
                ax.axis("off")
                continue
            ymax = 0.0
            for S, col, ls, name in ((Sg, G4_COL, "-", "Geant4"), (Sc, CC3_COL, "--", "CC3")):
                m = cell(S, reg, b)
                v = finite((S["dtheta"] if kind == "angle" else S["mass"])[m])
                n = len(v)
                if n == 0:
                    continue
                if n < args.min_n:
                    print("WARNING %s %s %s %s GeV: only %d pairs" % (kind, name, reg, E_NAMES[b], n))
                c, _ = np.histogram(v, bins=edges)
                frac = c / n
                ymax = max(ymax, float(frac.max()))
                lab = ("%s  N=%d  median=%.2f mrad" % (name, n, np.median(v))) if kind == "angle" else "%s  N=%d" % (name, n)
                ax.step(edges, np.append(frac, frac[-1]), where="post", color=col, ls=ls, lw=1.6, label=lab)
                if kind == "angle":
                    rows.append([reg, E_NAMES[b], name, n, float(np.median(v)), float(np.mean(v))])
                else:
                    lo_, hi_ = 1000 * (M_PI0 - args.maxdelta), 1000 * (M_PI0 + args.maxdelta)
                    rows.append([reg, E_NAMES[b], name, n, int((v < lo_).sum()), int(((v >= lo_) & (v <= hi_)).sum()),
                                 int((v > hi_).sum()), float(np.median(v))])
            tops.setdefault(ci, []).append(ymax)
            if kind == "mass":
                for x in (1000 * (M_PI0 - args.maxdelta), 1000 * (M_PI0 + args.maxdelta)):
                    ax.axvline(x, color="gray", ls="--", lw=1)
            ax.set_xlim(edges[0], edges[-1])
            ax.legend(frameon=False, fontsize=8, loc="upper right")
            clean(ax)
            if ri == 1:
                ax.set_xlabel("$\\alpha_{reco}-\\alpha_{true}$ [mrad]" if kind == "angle" else "two-photon mass [MeV]")
            if ci == 0:
                ax.set_ylabel("fraction of pairs per %g mrad bin" % args.angle_bin if kind == "angle" else "fraction of pairs per 5 MeV bin")
    if tops:
        allmax = max(max(v) for v in tops.values())
        for ci in tops:
            top = 1.3 * (allmax if kind == "angle" else max(tops[ci]))
            if top <= 0:
                top = 1.0
            for ri in range(2):
                axes[ri][ci].set_ylim(0, top)
    if kind == "angle":
        fig.suptitle("Opening-angle error $\\alpha_{reco}-\\alpha_{true}$ of resolved $\\pi^0$ pairs (fraction of pairs per %g mrad bin)" % args.angle_bin, fontsize=12)
    else:
        fig.suptitle("Reconstructed two-photon mass before the finder (fraction of pairs per 5 MeV bin, dashed lines = mass-window edges)", fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.08, hspace=0.45, wspace=0.18)
    name = "plot2_opening_angle" if kind == "angle" else "plot4_mass"
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=args.dpi)
    plt.close(fig)
    if kind == "angle":
        write_csv(os.path.join(outdir, name + ".csv"), ["region", "true_E_GeV", "file", "N", "median_mrad", "mean_mrad"], rows)
    else:
        write_csv(os.path.join(outdir, name + ".csv"),
                  ["region", "true_E_GeV", "file", "N", "below_window", "inside_window", "above_window", "median_MeV"], rows)


# ------------------------------------------------------------------ plot 3
def plot3(Sg, Sc, args, outdir):
    xc = centres()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col", sharey="row")
    rows = []
    for ci, reg in enumerate(REGIONS):
        axt, axr = axes[0][ci], axes[1][ci]
        med = {"G4": [], "CC3": []}
        txt = []
        for b in range(len(E_NAMES)):
            for S, tag in ((Sg, "G4"), (Sc, "CC3")):
                m = cell(S, reg, b)
                v = finite(np.concatenate([S["r0"][m], S["r1"][m]]))
                if len(v) < args.min_n:
                    print("WARNING plot3 %s %s %s GeV: only %d photons, point not drawn" % (tag, reg, E_NAMES[b], len(v)))
                med[tag].append(float(np.median(v)) if len(v) >= args.min_n else float("nan"))
                rows.append([reg, E_NAMES[b], tag, len(v), med[tag][-1]])
            txt.append(pct(photon_share(Sc, cell(Sc, reg, b))))
        mg, mc = np.array(med["G4"]), np.array(med["CC3"])
        ratio = mc / mg
        axt.plot(xc, mg, "-o", color=G4_COL, lw=1.6, label="Geant4")
        axt.plot(xc, mc, "--s", color=CC3_COL, lw=1.6, label="CC3")
        axr.plot(xc, ratio, "-o", color="black", lw=1.2, ms=5)
        axr.axhline(1.0, color="gray", lw=1)
        axt.set_ylim(*args.resp_ylim)
        axr.set_ylim(*args.ratio_ylim)
        for nm, arr, lim in (("G4", mg, args.resp_ylim), ("CC3", mc, args.resp_ylim), ("ratio", ratio, args.ratio_ylim)):
            bad = arr[np.isfinite(arr) & ((arr < lim[0]) | (arr > lim[1]))]
            if bad.size:
                print("WARNING plot3 %s %s: %d point(s) outside fixed y range %s" % (reg, nm, bad.size, list(lim)))
        set_energy_axis(axr)
        axt.set_title(reg)
        if ci == 0:
            axt.legend(frameon=False)
            axt.set_ylabel("median  $E_{reco}/E_{true}$")
            axr.set_ylabel("CC3 / G4")
        axr.set_xlabel("true $\\pi^0$ energy [GeV]")
        strip(axr, xc, txt, -0.62, "black", "CC3 fast-sim photons [%]" if ci == 0 else "")
        clean(axt); clean(axr)
    fig.suptitle("Photon energy response: median $E_{reco}/E_{true}$ vs true $\\pi^0$ energy (both photons of each resolved pair)", fontsize=11)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.2, hspace=0.12, wspace=0.12)
    fig.savefig(os.path.join(outdir, "plot3_energy_response.png"), dpi=args.dpi)
    plt.close(fig)
    write_csv(os.path.join(outdir, "plot3_energy_response.csv"), ["region", "true_pi0_E_GeV", "file", "N_photons", "median_E_reco_over_E_true"], rows)


# ------------------------------------------------------------------ plot 5
def photon_theta(S):
    a, g = S["a"], S["gen"]
    th, fl = [], []
    for j in (0, 1):
        sel = g & (a["mc_ph%d_E" % j] >= E_PHOTON_MIN)
        th.append(a["mc_ph%d_theta" % j][sel].astype(float))
        fl.append(a["mc_ph%d_fast" % j][sel] == 1)
    return np.concatenate(th), np.concatenate(fl)


def plot5(Sg, Sc, args, outdir):
    edges = np.arange(0.0, 180.0 + args.theta_bin, args.theta_bin)
    tg, _ = photon_theta(Sg)
    tc, fc = photon_theta(Sc)
    cg, _ = np.histogram(tg, bins=edges)
    cc, _ = np.histogram(tc, bins=edges)
    cf, _ = np.histogram(tc[fc], bins=edges)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.fill_between(edges, np.append(cf, cf[-1]), step="post", color=CC3_COL, alpha=0.25, lw=0, label="CC3 fast-simulated")
    ax.step(edges, np.append(cg, cg[-1]), where="post", color=G4_COL, lw=1.6, label="Geant4")
    ax.step(edges, np.append(cc, cc[-1]), where="post", color=CC3_COL, lw=1.6, ls="--", label="CC3")
    for x in (15, 30, 150, 165):
        ax.axvline(x, color="black", ls="--", lw=1)
    ax.set_xlim(0, 180)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("true photon theta [deg]")
    ax.set_ylabel("photons per %g deg" % args.theta_bin)
    ax.set_title("True polar angle of photons from generator $\\pi^0$ with E \u2265 10 GeV (photons per %g deg bin; dashed lines at 15, 30, 150, 165 deg)" % args.theta_bin, fontsize=11)
    ax.legend(frameon=False)
    clean(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "plot5_theta.png"), dpi=args.dpi)
    plt.close(fig)
    write_csv(os.path.join(outdir, "plot5_theta.csv"), ["theta_lo", "theta_hi", "G4_photons", "CC3_photons", "CC3_flagged"],
              [[edges[i], edges[i + 1], int(cg[i]), int(cc[i]), int(cf[i])] for i in range(len(cg))])


# ------------------------------------------------------------------ checks and notes
def self_checks(Sg, Sc, args):
    print("SELF-CHECKS (read these first)")
    for S in (Sg, Sc):
        a = S["a"]
        n_fl = int(((a["mc_ph0_fast"] == 1) | (a["mc_ph1_fast"] == 1))[S["gen"]].sum())
        print("  %-4s generator pi0s %d | pi0s with a flagged photon %d%s | res_E pairs %d | found %d" %
              (S["tag"], S["gen"].sum(), n_fl, " (G4 must be 0)" if S["tag"] == "G4" else "", S["res_E"].sum(), (S["gen"] & S["found"]).sum()))
        fo = S["found"] & np.isfinite(S["mass"]) & ((S["mass"] < 1000 * (M_PI0 - args.maxdelta)) | (S["mass"] > 1000 * (M_PI0 + args.maxdelta)))
        print("  %-4s found pairs outside the window given (+-%.0f MeV): %d (must be 0 if --maxdelta matches the finder run)" %
              (S["tag"], 1000 * args.maxdelta, fo.sum()))
    exp = LOG_EXPECT.get(args.suffix)
    if exp and abs(args.floor - 0.5) < 1e-9:
        for S in (Sg, Sc):
            got = (int(S["gen"].sum()), int((S["gen"] & S["found"]).sum()))
            print("  %-4s vs findings log: N_true/found %s, log %s -> %s" % (S["tag"], got, exp[S["tag"]], "OK" if got == exp[S["tag"]] else "DIFFERENT"))
    else:
        print("  no findings-log reference for suffix '%s' (log check only for full and full-dmgg0.1 at floor 0.5)" % args.suffix)
    print("  pairs per cell (res_E), G4 / CC3:")
    for reg in REGIONS:
        print("   %-7s" % reg + "  ".join("%s: %d/%d" % (E_NAMES[b], cell(Sg, reg, b).sum(), cell(Sc, reg, b).sum()) for b in range(len(E_NAMES))))
    print("WARNINGS from the plots follow below (if any)")


NOTES = """# Notes for the five plots (for you, not for the slides)

Sample for plots 1-4: generator pi0s (both photons genStatus 1) with two separate photon PFOs (best PFO per photon, cw >= %(floor)s, both PDG 22),
both PFO energies >= 0.5 GeV, no mass window. Energy axis: TRUE pi0 energy, bins 5-10, 10-20, 20-30, 30-50, 50-125 GeV.
Barrel: both photons at theta 40-140 deg and outside the phi cracks. Endcap (%(endcap)s): %(endcap_text)s.
Fast-sim numbers: CC3 file only. full = both photons flagged, partial = exactly one flagged. A flag needs a photon of about 10 GeV or more, so full is 0 at low energy by construction.

1. Mass-window acceptance ratio: for each file, resolved pairs with the unfitted mass inside 134.98 +- %(md)s MeV divided by all resolved pairs of the sample; the plot shows CC3 over G4.
   It is a conditional acceptance among resolved pairs (pairs lost earlier, e.g. merged or weakly linked, are not in the denominator), not an efficiency for finding all pi0s.
   Error bars: binomial error of each fraction, combined for the ratio. G4 and CC3 share generator events, so the bars are approximate. No significance is quoted.
2. Opening-angle error: alpha_reco is the angle between the two photon PFO momenta, alpha_true between the two generator photons.
   y = fraction of all pairs of that file, region and energy slice per bin. All six panels share one x range, one bin width (%(ab)s mrad) and one y axis, so bar heights are directly comparable.
   Pairs outside the drawn x-range are not drawn; N and the median in the legend use ALL pairs. alpha is positive by definition, so direction noise alone raises the mean of alpha_reco.
3. Energy response: E_reco / E_true per photon, both daughters of each pair pooled, binned by TRUE pi0 energy, median per bin (the fractional photon energy enters the mass as dm/m = 0.5 dE1/E1 + 0.5 dE2/E2 + dalpha/alpha).
   Fixed y axes (top %(rylim)s, ratio %(qlim)s), shared between barrel and endcap. No mass window. No error bars; check N in plot3_energy_response.csv.
4. Mass: invariant mass of the two photon PFOs before any finder fit. Dashed lines are the window edges. y = fraction of pairs per 5 MeV bin, shared within a column. Below/inside/above counts are in plot4_mass.csv.
   Pairs above 260 MeV are not drawn.
5. Theta: photons of generator pi0s with true energy >= 10 GeV. Geant4 and CC3 use the same generator events, so the two lines should coincide (a few photons per bin may differ); the shaded area is the CC3 photons the flag marks as fast-simulated.
   Dashed lines at 15, 30, 150, 165 deg mark the endcap interior used in plots 1-4.
Status: all numbers come from the NPZ tables; no systematics; no Geant4-vs-Geant4 baseline.
"""


def write_notes(args, outdir):
    et = ("both photons in folded theta [15,30) deg and on the same side, all phi" if args.endcap == "both_photons"
          else "pi0 direction in folded theta [15,30) deg, both photons on the same side")
    with open(os.path.join(outdir, "notes.md"), "w") as f:
        f.write(NOTES % dict(floor=args.floor, endcap=args.endcap, endcap_text=et, md="%.0f" % (1000 * args.maxdelta), ab="%g" % args.angle_bin,
                             rylim="%g-%g" % tuple(args.resp_ylim), qlim="%g-%g" % tuple(args.ratio_ylim)))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Five presentation plots, G4 vs CC3, barrel vs endcap.")
    ap.add_argument("--indir", default=REC_DIR)
    ap.add_argument("--suffix", default="full")
    ap.add_argument("--maxdelta", type=float, default=None)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--endcap", choices=["both_photons", "pi0_dir"], default="both_photons")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--angle-bin", type=float, default=0.25, help="bin width of the opening-angle error histograms [mrad]")
    ap.add_argument("--nbins", type=int, default=30, help="unused, kept so old command lines still work")
    ap.add_argument("--theta-bin", type=float, default=1.0)
    ap.add_argument("--resp-ylim", type=float, nargs=2, default=[0.98, 1.06])
    ap.add_argument("--ratio-ylim", type=float, nargs=2, default=[0.95, 1.05])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args(argv)
    if a.maxdelta is None:
        a.maxdelta = 0.1 if "dmgg0.1" in a.suffix else 0.04
    return a


def main(argv=None):
    args = parse_args(argv)
    outdir = args.outdir or os.path.join(PLOT_DIR, "endcap-plots-%s" % args.suffix)
    os.makedirs(outdir, exist_ok=True)
    sys.stdout = Tee(os.path.join(outdir, "summary.txt"))
    try:
        gp = os.path.join(args.indir, "pi0-tables-g4-%s.npz" % args.suffix)
        cp = os.path.join(args.indir, "pi0-tables-cc3-%s.npz" % args.suffix)
        print("endcap_plots_nov9.py v2 | %s , %s | window +-%.3f GeV | floor %.2f | endcap option %s | out %s" %
              (gp, cp, args.maxdelta, args.floor, args.endcap, outdir))
        need = args.endcap == "pi0_dir"
        Sg = build_state(load_tables(gp, need), "G4", args.floor, args.maxdelta, args.endcap)
        Sc = build_state(load_tables(cp, need), "CC3", args.floor, args.maxdelta, args.endcap)
        self_checks(Sg, Sc, args)
        plot1(Sg, Sc, args, outdir)
        grid_plot("angle", Sg, Sc, args, outdir)
        plot3(Sg, Sc, args, outdir)
        grid_plot("mass", Sg, Sc, args, outdir)
        plot5(Sg, Sc, args, outdir)
        write_notes(args, outdir)
        print("\nDone. Output folder: %s" % outdir)
    finally:
        sys.stdout = sys.__stdout__
    return outdir


if __name__ == "__main__":
    main()
