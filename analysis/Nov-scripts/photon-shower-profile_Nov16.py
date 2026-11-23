#!/usr/bin/env python3
"""photon_shower_profile_study.py  (simple version)

Reconstructed photon cluster width, centroid offset and direct-match fraction,
Geant4 vs CaloClouds3 (CC3), from the November 6 NPZ triplets.

Run:  python3 -u photon_shower_profile_study.py
      (optional: --input DIR  --outdir DIR)

Definitions: see README.md written to the output folder. Follows V6 / findings log.
Needs Python 3.9+, NumPy, Matplotlib.
"""
import argparse
import csv
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

DEFAULT_INPUT = "/data/dust/user/alimuham/thesis/reco/npz-data-summary/full"
DEFAULT_OUT = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/photon_shower_profile_study"

# ---- established conventions (V6 / findings log) ----------------------------
E_EDGES = np.array([10, 20, 30, 40, 60, 80, 100, 125], dtype=float)   # V6 E_BINS, lower edge inclusive
N_BINS = len(E_EDGES) - 1
E_CENTRES = np.sqrt(E_EDGES[:-1] * E_EDGES[1:])                         # V6 geometric centres
WEIGHT_CUT = 0.5
BARREL_THETA = (40.0, 140.0)
ENDCAP_FOLDED = (15.0, 30.0)                                            # folded theta, [lo, hi)
CRACK_CENTRES = -157.5 + 45.0 * np.arange(8)
DEAD_LO, DEAD_HI = -0.51, 8.5                                           # V6: (-0.01-0.5, 8+0.5)
SOLID_N, SOLID_EVENTS = 20, 2        # points below this are drawn open (never hidden)

POP_ALL, POP_PI0 = "All selected photons", "π⁰ daughter photons"
POPS = (POP_ALL, POP_PI0)
REGIONS = ("Full detector", "Barrel", "Endcap")
G4, HYB, FAST, UNFL, UNK = "G4", "CC3 hybrid", "CC3 fast", "CC3 unflagged", "CC3 unknown"
SHAPE_Q = ("width_ip", "width_centered", "centroid_offset")
QUANTITIES = ("owned_fraction",) + SHAPE_Q

STYLE = {
    G4:   dict(color="#2166ac", marker="o", ls="-",  label="Geant4"),
    HYB:  dict(color="#d6604d", marker="s", ls="-",  label="CC3 inclusive"),
    FAST: dict(color="#e08214", marker="^", ls="--", label="CC3 flagged"),
    UNFL: dict(color="#7f7f7f", marker="x", ls="--", label="CC3 unflagged"),
}
REGION_LABEL = {
    "Barrel": "Barrel interior — θ = 40°–140°",
    "Endcap": "Endcap interior — θ = 15°–30° and 150°–165°",
    "Full detector": "Full detector — baseline selection, no angular cut",
}
POP_MATH = {POP_ALL: "All selected photons", POP_PI0: r"$\pi^0$ daughter photons"}

plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.labelsize": 13,
                     "axes.titlesize": 12, "legend.fontsize": 11, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


# ---- helpers -----------------------------------------------------------------
_CID = {}   # collection IDs can be large hashes: map each distinct ID to a small code


def cid_code(cid):
    cid = np.asarray(cid, np.int64)
    uniq, inv = np.unique(cid, return_inverse=True)
    codes = np.array([_CID.setdefault(int(u), len(_CID)) for u in uniq], np.int64)
    return codes[inv]


def pack(loop, cid, idx):
    loop, idx = np.asarray(loop, np.int64), np.asarray(idx, np.int64)
    code = cid_code(cid)
    if len(loop) and (loop.min() < 0 or idx.min() < 0 or loop.max() >= 1 << 22
                      or idx.max() >= 1 << 24 or code.max() >= 1 << 16):
        raise ValueError(f"object key outside packing range: loop max {loop.max()}, "
                         f"idx max {idx.max()}, distinct collections {len(_CID)}")
    return (loop << 40) | (code << 24) | idx


def make_index(keys):
    keys = np.asarray(keys, np.int64)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    if len(sk) > 1 and np.any(sk[1:] == sk[:-1]):
        raise ValueError("duplicate object keys")

    def find(q):
        q = np.asarray(q, np.int64)
        if not len(sk):
            return np.full(len(q), -1, np.int64)
        i = np.clip(np.searchsorted(sk, q), 0, len(sk) - 1)
        return np.where(sk[i] == q, order[i], -1)
    return find


def in_dead_window(phi):
    phi = np.asarray(phi, float)
    dead = np.zeros(phi.shape, bool)
    for c in CRACK_CENTRES:
        dead |= ((phi - c - DEAD_LO) % 360.0) <= (DEAD_HI - DEAD_LO)
    return dead


def region_masks(theta, phi):
    fold = np.minimum(theta, 180.0 - theta)
    return {"Full detector": np.ones(len(theta), bool),
            "Barrel": (theta >= BARREL_THETA[0]) & (theta <= BARREL_THETA[1]) & ~in_dead_window(phi),
            "Endcap": (fold >= ENDCAP_FOLDED[0]) & (fold < ENDCAP_FOLDED[1])}


def add_moments(state, values, events):
    """state = [n, n_events, sum x, sum s_e^2, sum s_e n_e, sum n_e^2]; additive over chunks."""
    values = np.asarray(values, float)
    events = np.asarray(events)
    ok = np.isfinite(values)
    values, events = values[ok], events[ok]
    if not len(values):
        return
    _, inv = np.unique(events, return_inverse=True)
    s = np.bincount(inv, weights=values)
    n = np.bincount(inv).astype(float)
    state += np.array([len(values), len(s), values.sum(), s @ s, s @ n, n @ n])


def finish(state):
    """Mean over photons and its event-level (cluster-robust) standard error."""
    n, M, tot, s2, sn, n2 = (float(x) for x in state)
    if n == 0:
        return dict(n=0, M=0, mean=np.nan, se=np.nan)
    mean = tot / n
    var = max(0.0, s2 - 2 * mean * sn + mean * mean * n2)
    se = np.sqrt(M / (M - 1) * var) / n if M > 1 else np.nan
    return dict(n=int(n), M=int(M), mean=mean, se=se)


def open_part(folder, part, group, manifest, stack):
    label = manifest["sample"]
    name = f"{label}_{group}_{part['part']:05d}.npz"
    rec = next(r for r in part["files"] if r["file"] == name)
    path = Path(folder) / name
    if path.stat().st_size != rec["bytes"]:
        raise ValueError(f"file size differs from manifest: {path}")
    obj = stack.enter_context(np.load(path, allow_pickle=False))
    if obj["meta_sample"].item() != label or obj["meta_schema_version"].item() != 1:
        raise ValueError(f"wrong sample or schema: {path}")
    for table, n in rec["rows"].items():
        for token in manifest["schemas"][table].split():
            if f"{table}_{token.split(':')[0]}" not in obj.files:
                raise ValueError(f"missing field {table}_{token.split(':')[0]}: {path}")
        if len(obj[f"{table}_loop"]) != n:
            raise ValueError(f"row count mismatch in {table}: {path}")
    return obj


# ---- per-chunk physics ----------------------------------------------------------
def process_chunk(tr, rr, cl):
    C = Counter()
    C["events"] = len(tr["events_loop"])
    m = {k: tr["mc_" + k] for k in ("loop", "cid", "idx", "pdg", "gen", "fast", "E",
                                    "px", "py", "pz", "vx", "vy", "vz")}
    n_mc = len(m["loop"])
    find_mc = make_index(pack(m["loop"], m["cid"], m["idx"]))
    child = find_mc(pack(tr["ancestry_loop"], tr["ancestry_child_cid"], tr["ancestry_child_idx"]))
    parent = find_mc(pack(tr["ancestry_loop"], tr["ancestry_parent_cid"], tr["ancestry_parent_idx"]))
    if (child < 0).any() or (parent < 0).any():
        raise ValueError("ancestry refers to an unknown MC particle")

    is_ph = (m["pdg"] == 22) & (m["gen"] == 1)
    e_par = np.bincount(child, weights=(np.abs(m["pdg"][parent]) == 11).astype(float), minlength=n_mc) > 0
    n_dau = np.bincount(parent, minlength=n_mc)
    n_bad = np.bincount(parent, weights=(~is_ph[child]).astype(float), minlength=n_mc)
    pi0_ok = (m["pdg"] == 111) & (n_dau == 2) & (n_bad == 0)      # pi0 with exactly two genStatus-1 photon daughters
    is_dau = np.bincount(child, weights=pi0_ok[parent].astype(float), minlength=n_mc) > 0

    P = np.column_stack([m["px"], m["py"], m["pz"]])
    pmag = np.linalg.norm(P, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = P / pmag[:, None]
    theta = np.degrees(np.arctan2(np.hypot(m["px"], m["py"]), m["pz"]))
    phi = np.degrees(np.arctan2(m["py"], m["px"]))
    finite_dir = np.isfinite(u).all(1) & (pmag > 0)

    base = is_ph & ~e_par
    C["generator_photons"] = int(is_ph.sum())
    C["generator_photons_Ege10_no_e_parent"] = int((base & (m["E"] >= 10)).sum())
    C["E_ge_125_dropped"] = int((base & (m["E"] >= E_EDGES[-1])).sum())
    sel = base & (m["E"] >= E_EDGES[0]) & (m["E"] < E_EDGES[-1])
    C["invalid_truth_direction"] = int((sel & ~finite_dir).sum())
    rows = np.flatnonzero(sel & finite_dir)
    C["selected_photons"] = len(rows)
    for v, n in zip(*np.unique(m["fast"][rows], return_counts=True)):
        C[f"selected_fast_{int(v)}"] += int(n)
    if not len(rows):
        return None, C

    # direct ownership (V6): a photon PFO is owned by its largest cluster-weight link
    # (first stored link on a tie) if that weight >= 0.5; truth photon -> highest-energy owned PFO
    pf = {k: rr["pfo_" + k] for k in ("loop", "cid", "idx", "pdg", "E")}
    n_pfo = len(pf["loop"])
    find_pfo = make_index(pack(pf["loop"], pf["cid"], pf["idx"]))
    lp = find_pfo(pack(rr["links_loop"], rr["links_pfo_cid"], rr["links_pfo_idx"]))
    lm = find_mc(pack(rr["links_loop"], rr["links_mc_cid"], rr["links_mc_idx"]))
    if (lp < 0).any() or (lm < 0).any():
        raise ValueError("truth link refers outside the stored collections")
    cw = (np.rint(rr["links_raw"]).astype(np.int64) // 10000) / 1000.0
    if not np.allclose(cw, rr["links_cw"], atol=1e-9):
        raise ValueError("stored cluster weight disagrees with decoded raw weight")
    keep = pf["pdg"][lp] == 22
    lp, lm, cw = lp[keep], lm[keep], cw[keep]
    owner = np.full(n_mc, -1, np.int64)
    if len(lp):
        o = np.lexsort((np.arange(len(lp)), -cw, lp))
        lp_s, lm_s, cw_s = lp[o], lm[o], cw[o]
        top = np.r_[True, lp_s[1:] != lp_s[:-1]]
        bp, bm, bc = lp_s[top], lm_s[top], cw_s[top]
        own = bc >= WEIGHT_CUT
        bp, bm = bp[own], bm[own]
        if len(bp):
            o2 = np.lexsort((bp, -pf["E"][bp], bm))
            bp2, bm2 = bp[o2], bm[o2]
            f2 = np.r_[True, bm2[1:] != bm2[:-1]]
            owner[bm2[f2]] = bp2[f2]

    # first cluster of the owned PFO
    c = {k: cl["cluster_" + k] for k in ("loop", "cid", "idx", "n_hits", "hit_E_sum", "mean_x", "mean_y",
                                          "mean_z", "mean_xx", "mean_yy", "mean_zz", "mean_xy", "mean_xz", "mean_yz")}
    find_cl = make_index(pack(c["loop"], c["cid"], c["idx"]))
    s0 = rr["pfo_cluster_order"] == 0
    pr = find_pfo(pack(rr["pfo_cluster_loop"][s0], rr["pfo_cluster_pfo_cid"][s0], rr["pfo_cluster_pfo_idx"][s0]))
    cr = find_cl(pack(rr["pfo_cluster_loop"][s0], rr["pfo_cluster_cluster_cid"][s0], rr["pfo_cluster_cluster_idx"][s0]))
    if (pr < 0).any() or (cr < 0).any():
        raise ValueError("dangling PFO-cluster reference")
    if len(np.unique(pr)) != len(pr):
        raise ValueError("a PFO has two first clusters")
    first_cluster = np.full(n_pfo, -1, np.int64)
    first_cluster[pr] = cr

    own_pfo = owner[rows]
    owned = own_pfo >= 0
    crow = np.where(owned, first_cluster[np.maximum(own_pfo, 0)], -1)
    C["owned"] = int(owned.sum())
    C["owned_without_cluster"] = int((owned & (crow < 0)).sum())

    n = len(rows)
    w_ip, w_c, off, ident = (np.full(n, np.nan) for _ in range(4))
    k = np.flatnonzero(crow >= 0)
    if len(k):
        r = crow[k]
        cen = np.column_stack([c["mean_x"][r], c["mean_y"][r], c["mean_z"][r]])
        S = np.zeros((len(k), 3, 3))
        for (i, j), name in (((0, 0), "mean_xx"), ((1, 1), "mean_yy"), ((2, 2), "mean_zz"),
                             ((0, 1), "mean_xy"), ((0, 2), "mean_xz"), ((1, 2), "mean_yz")):
            S[:, i, j] = c[name][r]
            S[:, j, i] = c[name][r]
        uu = u[rows[k]]
        vv = np.column_stack([m["vx"][rows[k]], m["vy"][rows[k]], m["vz"][rows[k]]])
        ok = np.isfinite(cen).all(1) & np.isfinite(S).all((1, 2)) & np.isfinite(uu).all(1) & (c["hit_E_sum"][r] > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cmag = np.linalg.norm(cen, axis=1)
            a = cen / cmag[:, None]
            ip2 = np.trace(S, axis1=1, axis2=2) - np.einsum("ni,nij,nj->n", a, S, a)
            cov = S - cen[:, :, None] * cen[:, None, :]
            vc = np.trace(cov, axis1=1, axis2=2) - np.einsum("ni,nij,nj->n", uu, cov, uu)
            d = cen - vv
            off2 = (d * d).sum(1) - ((d * uu).sum(1)) ** 2
            Sv = S - cen[:, :, None] * vv[:, None, :] - vv[:, :, None] * cen[:, None, :] + vv[:, :, None] * vv[:, None, :]
            vt = np.trace(Sv, axis1=1, axis2=2) - np.einsum("ni,nij,nj->n", uu, Sv, uu)
        ok_ip = ok & (c["n_hits"][r] >= 2) & (cmag > 0) & (ip2 >= -1e-6)
        okc = ok & (vc >= -1e-6)
        w_ip[k] = np.where(ok_ip, np.sqrt(np.maximum(ip2, 0)), np.nan)
        w_c[k] = np.where(okc, np.sqrt(np.maximum(vc, 0)), np.nan)
        off[k] = np.where(ok, np.sqrt(np.maximum(off2, 0)), np.nan)
        ident[k] = np.where(okc & (vt >= -1e-6), vt - vc - off2, np.nan)
    C["negative_variance_ip"] = int((np.isnan(w_ip) & (crow >= 0) & np.isfinite(off) & (c["n_hits"][np.maximum(crow, 0)] >= 2)).sum())
    C["identity_checked"] = int(np.isfinite(ident).sum())
    C["identity_failed"] = int((np.abs(ident[np.isfinite(ident)]) > 1e-5).sum())

    rm = region_masks(theta[rows], phi[rows])
    table = dict(loop=m["loop"][rows], mc_cid=m["cid"][rows], mc_idx=m["idx"][rows], E=m["E"][rows],
                 theta=theta[rows], phi=phi[rows], fast=m["fast"][rows], is_dau=is_dau[rows], owned=owned,
                 width_ip=w_ip, width_centered=w_c, centroid_offset=off,
                 barrel=rm["Barrel"], endcap=rm["Endcap"])
    return table, C


# ---- aggregation -------------------------------------------------------------------
def group_masks(sample, fast):
    if sample == "g4":
        return {G4: np.ones(len(fast), bool)}
    return {HYB: np.ones(len(fast), bool), FAST: fast == 1, UNFL: fast == 0,
            UNK: (fast != 0) & (fast != 1)}


def accumulate(acc, table, sample):
    b = np.searchsorted(E_EDGES, table["E"], side="right") - 1
    regs = {"Full detector": np.ones(len(b), bool), "Barrel": table["barrel"], "Endcap": table["endcap"]}
    pops = {POP_ALL: np.ones(len(b), bool), POP_PI0: table["is_dau"]}
    for reg, rmask in regs.items():
        for pop, pmask in pops.items():
            for grp, gmask in group_masks(sample, table["fast"]).items():
                base = rmask & pmask & gmask
                for ib in np.unique(b[base]):
                    sel = base & (b == ib)
                    cell = acc.setdefault((reg, pop, grp, int(ib)),
                                          dict(n_truth=0, n_owned=0, m={q: np.zeros(6) for q in QUANTITIES}))
                    cell["n_truth"] += int(sel.sum())
                    cell["n_owned"] += int(table["owned"][sel].sum())
                    add_moments(cell["m"]["owned_fraction"], table["owned"][sel].astype(float), table["loop"][sel])
                    so = sel & table["owned"]
                    for q in SHAPE_Q:
                        add_moments(cell["m"][q], table[q][so], table["loop"][so])


def build_results(acc):
    res = {}
    empty = dict(n_truth=0, n_owned=0, n_valid=0, n_events=0, mean=np.nan, se=np.nan)
    for reg in REGIONS:
        for pop in POPS:
            for grp in (G4, HYB, FAST, UNFL, UNK):
                for q in QUANTITIES:
                    series = []
                    for ib in range(N_BINS):
                        cell = acc.get((reg, pop, grp, ib))
                        if cell is None:
                            series.append(dict(empty))
                            continue
                        f = finish(cell["m"][q])
                        series.append(dict(n_truth=cell["n_truth"], n_owned=cell["n_owned"], n_valid=f["n"],
                                           n_events=f["M"], mean=f["mean"], se=f["se"]))
                    res[(reg, pop, grp, q)] = series
    return res


def write_tables(res, outdir):
    with (Path(outdir) / "photon_profile_bins.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "population", "sample", "energy_lo_GeV", "energy_hi_GeV", "quantity",
                    "n_truth_denominator", "n_owned_numerator", "n_valid", "n_events", "mean", "se_event",
                    "low_statistics", "unit"])
        for (reg, pop, grp, q), series in res.items():
            for ib, s in enumerate(series):
                if s["n_truth"]:
                    low = int(s["n_valid"] < SOLID_N or s["n_events"] < SOLID_EVENTS)
                    w.writerow([reg, pop, grp, E_EDGES[ib], E_EDGES[ib + 1], q, s["n_truth"], s["n_owned"],
                                s["n_valid"], s["n_events"], s["mean"], s["se"], low,
                                "fraction(0-1)" if q == "owned_fraction" else "mm"])
    with (Path(outdir) / "cc3_minus_g4.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "population", "cc3_group", "energy_lo_GeV", "energy_hi_GeV", "quantity", "g4_mean",
                    "cc3_mean", "difference", "relative_difference_percent", "se_difference_independent_samples",
                    "n_valid_g4", "n_valid_cc3", "note"])
        for reg in REGIONS:
            for pop in POPS:
                for grp in (HYB, FAST, UNFL):
                    for q in ("width_ip", "centroid_offset"):
                        for ib in range(N_BINS):
                            a, b = res[(reg, pop, G4, q)][ib], res[(reg, pop, grp, q)][ib]
                            if np.isfinite(a["mean"]) and np.isfinite(b["mean"]):
                                d = b["mean"] - a["mean"]
                                w.writerow([reg, pop, grp, E_EDGES[ib], E_EDGES[ib + 1], q, a["mean"], b["mean"], d,
                                            100 * d / a["mean"] if a["mean"] else np.nan, np.hypot(a["se"], b["se"]),
                                            a["n_valid"], b["n_valid"],
                                            "conditional on direct match; SE assumes independent samples (G4/CC3 share generator events)"])


# ---- plotting ----------------------------------------------------------------------
def draw(ax, series, grp, scale=1.0, dodge=0.0, fraction=False):
    st = STYLE[grp]
    x = E_CENTRES * (1.0 + dodge)
    nval = np.array([s["n_truth"] if fraction else s["n_valid"] for s in series])
    mean = np.array([s["mean"] for s in series]) * scale
    se = np.array([s["se"] for s in series]) * scale
    ok = (nval >= 1) & np.isfinite(mean)
    y = np.where(ok, mean, np.nan)
    solid = ok & (nval >= SOLID_N) & (np.array([s["n_events"] for s in series]) >= SOLID_EVENTS)
    ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.6, label=st["label"], zorder=2)
    for mask, hollow in ((solid, False), (ok & ~solid, True)):
        if mask.any():
            e = np.where(np.isfinite(se[mask]), se[mask], 0.0)
            kw = dict(color=st["color"], marker=st["marker"], ms=6.5, ls="none", capsize=3, elinewidth=1.2,
                      markeredgewidth=1.5, zorder=3)
            if hollow:
                kw.update(mfc="white", alpha=0.75)
            ax.errorbar(x[mask], y[mask], yerr=e, **kw)
    err0 = np.where(np.isfinite(se), se, 0)
    return np.where(ok, y - err0, np.nan), np.where(ok, y + err0, np.nan)


def energy_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(E_EDGES[0], E_EDGES[-1])
    ax.set_xticks([10, 20, 30, 50, 100, 125])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis="y", alpha=0.25)


def legend(fig, ax, y):
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", marker="o", mfc="white", ls="none",
                          label=f"open: <{SOLID_N} photons or <{SOLID_EVENTS} events"))
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, y), fontsize=10)


def footer(breakdown):
    t = ("Mean over directly owned photons: PFO PDG 22 with largest cluster-weight link ≥ 0.5 to the truth photon; leading-energy PFO; first cluster. "
         "Truth photons: genStatus 1, 10 ≤ E < 125 GeV, no e± parent.\n"
         "Width = energy-weighted transverse RMS of cluster hits about the origin→cluster-centroid axis. Centroid offset = unsigned distance of the energy-weighted "
         "centroid from the true photon line (production vertex, true direction).\n"
         "Barrel: φ windows [C−0.51°, C+8.5°], C = −157.5° + 45°k, excluded; endcap: all φ. Error bars: event-level standard errors. "
         "Y-axes do not start at zero. Conditional on direct match.")
    if breakdown:
        t += ("\nFlagged/unflagged = stored fast-sim flag of the original MC photon (1/0); unflagged is NOT verified Geant4-only "
              "(many convert; descendants may be flagged).")
    return t


def shape_grid(res, cols, groups, fname, outdir, title, breakdown, extra=""):
    """cols: list of (region, population, column title)."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.6), sharex=True)
    quant = (("width_ip", "Mean transverse RMS width [mm]"), ("centroid_offset", "Mean centroid offset [mm]"))
    lims = [[np.inf, -np.inf], [np.inf, -np.inf]]
    for j, (reg, pop, ctitle) in enumerate(cols):
        for i, (q, ylab) in enumerate(quant):
            ax = axes[i, j]
            for g in groups:
                lo, hi = draw(ax, res[(reg, pop, g, q)], g)
                if np.isfinite(lo).any():
                    lims[i][0] = min(lims[i][0], np.nanmin(lo))
                    lims[i][1] = max(lims[i][1], np.nanmax(hi))
            ax.set_ylabel(ylab)
            ax.tick_params(labelleft=True)
            energy_axis(ax)
            if i == 0:
                ax.set_title(ctitle)
            else:
                ax.set_xlabel("True photon energy [GeV]")
    for i in range(2):
        lo, hi = lims[i]
        if np.isfinite(lo):
            pad = 0.08 * ((hi - lo) or abs(hi) or 1.0)
            for j in range(2):
                axes[i, j].set_ylim(max(0.0, lo - pad), hi + pad)
    legend(fig, axes[0, 0], 0.945)
    fig.suptitle(title, y=0.985, fontsize=14)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.2, hspace=0.12, wspace=0.17)
    fig.text(0.02, 0.012, footer(breakdown) + extra, fontsize=6.8, va="bottom", ha="left", color="#333333")
    fig.savefig(Path(outdir) / fname, dpi=170)
    plt.close(fig)


def fig_matching(res, outdir):
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 11.0), sharex=True, sharey=True)
    dodge = {G4: -0.045, HYB: -0.015, FAST: 0.015, UNFL: 0.045}
    for i, reg in enumerate(REGIONS):
        for j, pop in enumerate(POPS):
            ax = axes[i, j]
            for g in (G4, HYB, FAST, UNFL):
                draw(ax, res[(reg, pop, g, "owned_fraction")], g, scale=100.0, dodge=dodge[g], fraction=True)
            ax.set_ylim(-3, 103)
            ax.set_ylabel("Directly matched [%]")
            ax.tick_params(labelleft=True)
            energy_axis(ax)
            if i == 0:
                ax.set_title(POP_MATH[pop])
            if j == 1:
                ax.text(1.02, 0.5, REGION_LABEL[reg].split(" — ")[0], transform=ax.transAxes, rotation=270,
                        va="center", fontsize=11)
            if i == 2:
                ax.set_xlabel("True photon energy [GeV]")
    legend(fig, axes[0, 0], 0.955)
    fig.suptitle("Direct photon matching fraction", y=0.99, fontsize=15)
    fig.subplots_adjust(left=0.09, right=0.95, top=0.9, bottom=0.17, hspace=0.14, wspace=0.14)
    cap = ("Numerator: selected truth photons with ≥1 directly owned photon PFO (PDG 22, largest cluster-weight link to that truth photon, weight ≥ 0.5).\n"
           "Denominator: all selected truth photons in the bin (genStatus 1, 10 ≤ E < 125 GeV, no e± parent), including photons without any PFO.\n"
           "NOT the total photon reconstruction efficiency: converted photons can be reconstructed through descendants (e±) and count as unmatched here.\n"
           "Low direct-match rates of CC3 unflagged photons therefore do not show a failed Geant4-fallback reconstruction. "
           "Flag = original MC photon; unflagged is not verified Geant4-only.\n"
           "Barrel: φ cracks [C−0.51°, C+8.5°] excluded; endcap θ = 15°–30° and 150°–165°. Event-level standard errors; "
           "x positions offset by ≤4.5% for legibility; a zero fraction has zero plug-in error.")
    fig.text(0.02, 0.012, cap, fontsize=6.8, va="bottom", ha="left", color="#333333")
    fig.savefig(Path(outdir) / "06_direct_matching_fraction.png", dpi=170)
    plt.close(fig)


def fig_counts(res, outdir):
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.0), sharex=True)
    for i, reg in enumerate(REGIONS):
        for j, pop in enumerate(POPS):
            ax = axes[i, j]
            for g in (G4, HYB, FAST, UNFL):
                st = STYLE[g]
                nt = np.array([s["n_truth"] for s in res[(reg, pop, g, "owned_fraction")]], float)
                nv = np.array([s["n_valid"] for s in res[(reg, pop, g, "width_ip")]], float)
                ax.plot(E_CENTRES, np.where(nt > 0, nt, np.nan), color=st["color"], ls="-", marker=st["marker"], ms=5)
                ax.plot(E_CENTRES, np.where(nv > 0, nv, np.nan), color=st["color"], ls=":", marker=st["marker"], ms=5, mfc="white")
            ax.axhline(SOLID_N, color="0.6", lw=0.8, ls="-.")
            ax.set_yscale("log")
            energy_axis(ax)
            ax.set_ylabel("Photons per bin")
            if i == 0:
                ax.set_title(POP_MATH[pop])
            if j == 1:
                ax.text(1.02, 0.5, REGION_LABEL[reg].split(" — ")[0], transform=ax.transAxes, rotation=270,
                        va="center", fontsize=11)
            if i == 2:
                ax.set_xlabel("True photon energy [GeV]")
    hs = [Line2D([], [], color=STYLE[g]["color"], marker=STYLE[g]["marker"], ls="-", label=STYLE[g]["label"])
          for g in (G4, HYB, FAST, UNFL)]
    hs += [Line2D([], [], color="0.3", ls="-", label="selected truth photons (denominator)"),
           Line2D([], [], color="0.3", ls=":", marker="o", mfc="white", label="valid width_ip values"),
           Line2D([], [], color="0.6", ls="-.", label=f"{SOLID_N} photons")]
    fig.legend(hs, [h.get_label() for h in hs], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.96), fontsize=9)
    fig.suptitle("Supplement — raw photon counts per energy bin", y=0.995, fontsize=14)
    fig.subplots_adjust(left=0.09, right=0.95, top=0.88, bottom=0.07, hspace=0.12, wspace=0.2)
    fig.savefig(Path(outdir) / "S1_counts_per_bin.png", dpi=170)
    plt.close(fig)


def make_figures(res, out):
    barrel_endcap = lambda pop: [("Barrel", pop, REGION_LABEL["Barrel"]), ("Endcap", pop, REGION_LABEL["Endcap"])]
    ttl = lambda pop: f"Reconstructed photon cluster width and centroid offset — {POP_MATH[pop]}"
    shape_grid(res, barrel_endcap(POP_ALL), (G4, HYB), "01_all_photons_g4_vs_cc3.png", out, ttl(POP_ALL), False)
    shape_grid(res, barrel_endcap(POP_ALL), (G4, HYB, FAST, UNFL), "02_all_photons_cc3_breakdown.png", out, ttl(POP_ALL), True)
    shape_grid(res, barrel_endcap(POP_PI0), (G4, HYB), "03_pi0_daughters_g4_vs_cc3.png", out, ttl(POP_PI0), False)
    shape_grid(res, barrel_endcap(POP_PI0), (G4, HYB, FAST, UNFL), "04_pi0_daughters_cc3_breakdown.png", out, ttl(POP_PI0), True)
    shape_grid(res, [("Full detector", p, POP_MATH[p]) for p in POPS], (G4, HYB), "05_full_detector_summary.png", out,
               "Full detector — baseline photon selection, no angular-region cut", False,
               "\nFull detector includes forward, gap and crack regions where the calorimeter may not contain the shower; "
               "it does not mean every photon is reconstructed. Populations overlap (π⁰ daughters ⊂ all photons).")
    fig_matching(res, out)
    fig_counts(res, out)


README = """# photon_shower_profile_study

Run: python3 -u photon_shower_profile_study.py [--input DIR] [--outdir DIR]
Input: November 6 NPZ triplets (G4 and CC3 100k REC files), manifests in <input>/g4 and <input>/cc3.

Truth photons: PDG 22, genStatus 1, 10 <= E < 125 GeV, no parent with |PDG| = 11, finite direction.
pi0 daughters: subset whose parent is a PDG 111 with exactly two daughters, both genStatus-1 photons (overlaps the all-photon sample).
Bins: 10,20,30,40,60,80,100,125 GeV (V6 E_BINS), lower edge inclusive, geometric bin centres.
Barrel: 40 <= theta <= 140 deg, phi outside [C-0.51, C+8.5] deg, C = -157.5 + 45k.
Endcap: 15 <= min(theta, 180-theta) < 30 deg, all phi. Full detector: no angular cut. Regions overlap.
Matching (V6): photon PFO owned by its largest cluster-weight link if >= 0.5 (first stored link on a tie); truth photon takes its
highest-energy owned PFO and that PFO's first cluster. Direct-match fraction = owned / selected truth photons (not an efficiency).
width_ip: sqrt(sum E d^2 / sum E), d = distance of hit from the origin->energy-centroid axis (V6 sigma_perp), >= 2 hits.
width_centered: same about the line through the centroid parallel to the true direction.
centroid_offset: unsigned distance from the energy centroid to the line through the true vertex along the true direction.
Hits: all hits of the first cluster, raw signed energies as stored (ECAL-only membership not enforced, as in V6).
CC3 flagged/unflagged: stored flag of the original MC photon (1/0); unflagged is NOT verified Geant4-only.
Errors: event-level (cluster-robust) standard error. Open markers: <20 photons or <2 events; nothing is hidden.
Files: photon_profile_bins.csv, cc3_minus_g4.csv (SE assumes independent samples), validation_summary.txt, figures 01-06, S1.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    ap.add_argument("--outdir", type=Path, default=Path(DEFAULT_OUT))
    args = ap.parse_args()
    out = args.outdir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    lines, acc = [], {}
    for sample in ("g4", "cc3"):
        folder = args.input.expanduser() / sample
        mpath = next((p for p in (folder / "manifest.json", folder / f"{sample}_manifest.json") if p.is_file()), None)
        if mpath is None:
            raise FileNotFoundError(f"manifest not found in {folder}")
        man = json.loads(mpath.read_text())
        if man["sample"] != sample or man["schema_version"] != 1 or not man["complete"]:
            raise ValueError(f"unexpected manifest {mpath}")
        tot, nxt = Counter(), 0
        for part in man["chunks"]:
            if part["first_event_loop"] != nxt:
                raise ValueError("non-contiguous manifest")
            with ExitStack() as stack:
                tr, rr, cl = (open_part(folder, part, g, man, stack) for g in ("truth", "reco", "clusters"))
                if not np.array_equal(tr["events_loop"], np.arange(nxt, part["last_event_loop"] + 1)):
                    raise ValueError("unexpected event loops")
                for d in (rr, cl):
                    for fld in ("loop", "run", "event"):
                        if not np.array_equal(tr["events_" + fld], d["events_" + fld]):
                            raise ValueError("triplet event identity mismatch")
                table, C = process_chunk(tr, rr, cl)
            tot.update(C)
            if table is not None:
                accumulate(acc, table, sample)
            nxt = part["last_event_loop"] + 1
            print(f"{sample}: chunk {part['part'] + 1}/{len(man['chunks'])}, {nxt} events", flush=True)
        mc = man["counters"]
        for mine, theirs in (("events", "events_processed"), ("generator_photons", "generator_photons"),
                             ("generator_photons_Ege10_no_e_parent", "generator_photons_Ege10_no_e_parent")):
            lines.append(f"{'PASS' if tot[mine] == mc[theirs] else 'FAIL'}  {sample} closure {mine}: {tot[mine]} vs manifest {mc[theirs]}")
        lines.append(f"INFO  {sample}: selected {tot['selected_photons']}, owned {tot['owned']}, owned without first cluster "
                     f"{tot['owned_without_cluster']}, E>=125 dropped {tot['E_ge_125_dropped']}, invalid directions "
                     f"{tot['invalid_truth_direction']}, negative-variance width_ip {tot['negative_variance_ip']}")
        lines.append(f"INFO  {sample}: selected photons by stored fast flag: " +
                     ", ".join(f"{k.split('_')[-1]}: {v}" for k, v in sorted(tot.items()) if k.startswith("selected_fast_")))
        if sample == "g4":
            lines.append(f"{'PASS' if tot['selected_fast_1'] == 0 else 'FAIL'}  g4: flagged selected photons in the G4 file (expected 0): {tot['selected_fast_1']}")
        lines.append(f"{'PASS' if tot['identity_failed'] == 0 else 'FAIL'}  {sample} width identity: {tot['identity_checked']} checked, {tot['identity_failed']} failed (>1e-5 mm^2)")
    res = build_results(acc)
    write_tables(res, out)
    make_figures(res, out)
    (out / "validation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    print("\n".join(lines))
    print(f"done. output: {out}")


if __name__ == "__main__":
    main()
