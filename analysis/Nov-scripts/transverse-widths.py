#!/usr/bin/env python3
"""plot_width_profiles.py

Transverse-width figures (G4 vs CC3) drawn from photon_profile_bins.csv
(written by photon_shower_profile_study.py). Reads only the CSV: no NPZ access, runs in seconds.

Run:  python3 transverse-widths.py [path/to/photon_profile_bins.csv]
Output (same folder as the CSV):
  W1_width_barrel_endcap.png   rows: G4 vs CC3 | CC3 split by fast-sim flag; columns: barrel | endcap
  W2_width_full_detector.png   full detector: G4 vs CC3 on top, CC3 split by flag below
Population: all selected photons (change POPULATION to use the pi0-daughter subset).
Totals in the corner of each panel = photons in the plotted averages (directly matched, valid cluster), summed over all energy bins.
"""
import csv
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DEFAULT_CSV = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/photon_shower_profile_study_nov16/photon_profile_bins.csv"
#POPULATION = "All selected photons"
POPULATION = "\u03c0\u2070 daughter photons"
QUANTITY = "width_ip"

G4, HYB, FAST, UNFL = "G4", "CC3 hybrid", "CC3 fast", "CC3 unflagged"
STYLE = {
    G4:   dict(color="#2166ac", marker="o", ls="-",  label="Geant4", alpha=1.0, lw=1.8, z=3),
    HYB:  dict(color="#d6604d", marker="s", ls="-",  label="CC3 inclusive", alpha=1.0, lw=1.8, z=4),
    FAST: dict(color="#e08214", marker="^", ls="--", label="CC3 flagged", alpha=1.0, lw=1.8, z=6),
    UNFL: dict(color="#7f7f7f", marker="x", ls="--", label="CC3 unflagged", alpha=1.0, lw=1.6, z=5),
}
FAINT = dict(alpha=0.35, lw=1.2)          # CC3 inclusive in the split panels, so the flagged line stays visible
REGION_TITLE = {
    "Barrel": "Barrel interior — θ = 40°–140°",
    "Endcap": "Endcap interior — θ = 15°–30° and 150°–165°",
    "Full detector": "Full detector — baseline selection, no angular cut",
}
HEADROOM = 0.74       # data fill at most this fraction of the axis height, leaving room for the totals

plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
                     "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "figure.autolayout": False, "figure.constrained_layout.use": False})


def load(path):
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
            if r["population"] == POPULATION and r["quantity"] == QUANTITY]
    lo = sorted({float(r["energy_lo_GeV"]) for r in rows})
    hi = {float(r["energy_lo_GeV"]): float(r["energy_hi_GeV"]) for r in rows}
    edges = np.array(lo + [hi[lo[-1]]])
    data = {}
    for r in rows:
        i = lo.index(float(r["energy_lo_GeV"]))
        s = data.setdefault((r["region"], r["sample"]), dict(mean=np.full(len(lo), np.nan), se=np.full(len(lo), np.nan),
                                                              n=np.zeros(len(lo), int)))
        s["mean"][i] = float(r["mean"])
        s["se"][i] = float(r["se_event"])
        s["n"][i] = int(r["n_valid"])
    return edges, data


def curve(ax, x, s, grp, faint=False):
    st = dict(STYLE[grp])
    if faint:
        st.update(FAINT)
    ok = (s["n"] >= 1) & np.isfinite(s["mean"])
    y = np.where(ok, s["mean"], np.nan)
    e = np.where(np.isfinite(s["se"]), s["se"], 0.0)
    ax.plot(x, y, color=st["color"], ls=st["ls"], lw=st["lw"], alpha=st["alpha"], label=st["label"], zorder=st["z"])
    ax.errorbar(x[ok], y[ok], yerr=e[ok], color=st["color"], alpha=st["alpha"], marker=st["marker"], ms=7, ls="none",
                capsize=3, elinewidth=1.3, markeredgewidth=1.6, zorder=st["z"])
    return np.where(ok, y - e, np.nan), np.where(ok, y + e, np.nan)


def energy_axis(ax, edges):
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xticks([10, 20, 30, 50, 100, 125])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis="y", alpha=0.25)


def totals_box(ax, items):
    """items: [(group, label, total)] drawn top right, one line each, in the curve colours."""
    for k, (grp, label, total) in enumerate(items):
        ax.text(0.985, 0.975 - 0.075 * k, f"{label}: {total:,} photon{'' if total == 1 else 's'}", transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color=STYLE[grp]["color"])


def set_limits(axes, bounds):
    lo = min(np.nanmin(a) for a, b in bounds if np.isfinite(a).any())
    hi = max(np.nanmax(b) for a, b in bounds if np.isfinite(b).any())
    lo -= 0.06 * (hi - lo)
    top = lo + (hi - lo) / HEADROOM
    for ax in axes:
        ax.set_ylim(lo, top)


def wrap(text, width):
    return "\n".join(textwrap.fill(p, width) for p in text.split("\n"))


YLAB = "Mean transverse RMS width [mm]"
CAPTION = ("Mean energy-weighted transverse RMS width of the first cluster of directly matched photons (distance of cluster hits from the origin→centroid axis). "
           "Photons: genStatus 1, 10 ≤ E < 125 GeV, no e± parent.\n"
           "Photon totals (top right) = photons in the plotted averages (directly matched, valid cluster), summed over all energy bins. "
           "Error bars: event-level standard errors. Barrel excludes the φ cracks; endcap uses all φ. Y-axes do not start at zero.\n"
           "Flagged / unflagged = fast-sim flag of the original MC photon (1 / 0). Unflagged is not verified Geant4-only. "
           "Most unflagged photons have no direct PFO match, so the inclusive CC3 average is almost entirely flagged photons.")


def panel(ax, x, edges, data, reg, kind, bounds):
    """kind 'main': G4 vs CC3 inclusive.  kind 'split': G4, CC3 (faint), flagged, unflagged."""
    tot = lambda g: int(data[(reg, g)]["n"].sum())
    if kind == "main":
        for g in (G4, HYB):
            bounds.append(curve(ax, x, data[(reg, g)], g))
        ax.text(0.02, 0.975, "Geant4 vs CC3 (inclusive)", transform=ax.transAxes, va="top", ha="left", fontsize=12, color="#222222")
        totals_box(ax, [(G4, "Geant4", tot(G4)), (HYB, "CC3", tot(HYB))])
    else:
        bounds.append(curve(ax, x, data[(reg, G4)], G4))
        bounds.append(curve(ax, x, data[(reg, HYB)], HYB, faint=True))
        bounds.append(curve(ax, x, data[(reg, FAST)], FAST))
        bounds.append(curve(ax, x, data[(reg, UNFL)], UNFL))
        ax.text(0.02, 0.975, "CC3 split by fast-sim flag", transform=ax.transAxes, va="top", ha="left", fontsize=12, color="#222222")
        totals_box(ax, [(FAST, "CC3 flagged", tot(FAST)), (UNFL, "CC3 unflagged", tot(UNFL))])
    energy_axis(ax, edges)
    ax.set_ylabel(YLAB)
    ax.tick_params(labelleft=True)
    ax.set_xlabel("True photon energy [GeV]")
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)


def fig_main(edges, data, out):
    x = np.sqrt(edges[:-1] * edges[1:])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.4))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.215, hspace=0.3, wspace=0.17)
    bounds = []
    for i, kind in enumerate(("main", "split")):
        for j, reg in enumerate(("Barrel", "Endcap")):
            panel(axes[i, j], x, edges, data, reg, kind, bounds)
            if i == 0:
                axes[i, j].set_title(REGION_TITLE[reg], pad=10)
    set_limits(axes[0], bounds[:4])          # shared y range per row, so barrel and endcap can be compared
    set_limits(axes[1], bounds[4:])
    fig.suptitle("Transverse width of reconstructed photon clusters — Geant4 vs CC3", fontsize=15, y=0.975)
    fig.text(0.015, 0.012, wrap(CAPTION, 175), fontsize=8, va="bottom", ha="left", color="#333333")
    fig.savefig(out / "W1_width_barrel_endcap.png", dpi=170)
    return fig


def fig_full(edges, data, out):
    x = np.sqrt(edges[:-1] * edges[1:])
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 12.0))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.9, bottom=0.23, hspace=0.3)
    bounds = []
    for ax, kind in zip(axes, ("main", "split")):
        panel(ax, x, edges, data, "Full detector", kind, bounds)
    axes[0].set_title(REGION_TITLE["Full detector"], pad=10)
    set_limits([axes[0]], bounds[:2])
    set_limits([axes[1]], bounds[2:])
    fig.suptitle("Transverse width — full detector, Geant4 vs CC3", fontsize=14, y=0.975)
    fig.text(0.015, 0.012, wrap(CAPTION + "\nFull detector includes forward, gap and crack regions where the calorimeter may not contain the whole shower.", 125),
             fontsize=7.2, va="bottom", ha="left", color="#333333")
    fig.savefig(out / "W2_width_full_detector.png", dpi=170)
    return fig


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV).expanduser()
    edges, data = load(path)
    fig_main(edges, data, path.parent)
    fig_full(edges, data, path.parent)
    print("written to", path.parent)


if __name__ == "__main__":
    main()
