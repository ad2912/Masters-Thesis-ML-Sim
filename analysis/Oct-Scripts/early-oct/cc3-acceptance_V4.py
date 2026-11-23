"""
cc3_acceptance_v4.py
====================
Where does CaloClouds3 take over from Geant4? Fast-sim fraction of MC photons
versus theta, phi and energy, from the 100k CC3 SIM file. Feeds slide S7.

Selection (base selection for every plot here)
  MCParticles with PDG == 22, generatorStatus == 1, E >= 10 GeV,
  and NO direct parent with |PDG| == 11 (ISR / e+- photons removed).
  Flag per photon: p.isHandledByFastSim() (Key4hep nightlies).

Region definitions (McKeown thesis, DESY-THESIS-2024-008, sec. 9.2.3)
  Trigger edges in theta: barrel 40-140 deg; forward endcap theta < 33;
  backward endcap theta > 147. The 33-40 and 140-147 windows ("theta gaps") are
  excluded by the trigger. These are TRIGGER values: the thesis says the window
  covers the geometric transition plus the range used when training the model.
  Phi dead zone (barrel, eq. 9.1): (C - 0.01, C + 8) deg for
  C = -157.5 + 45 k, k = 0..7. One-sided.
  Endcap "core" for phi/energy plots: 12.5-33 and 147-167.5 deg
  (chosen clear of the rising outer edge, read off the theta plot).
  No lines are drawn for the outer endcap edges (about 10 and 170 deg).

Each plot removes the exclusion of the OTHER variable and shows its own
  theta plots : phi dead zone removed; theta gaps shown.
  phi plots   : theta gaps removed (3a) or excluded by the theta range (3b, 3c);
                phi dead zones shown, shaded.
  phi mask    : drops photons inside the dead window plus 0.5 deg margin on each
                side. Default: BARREL photons only (the thesis describes it for
                the barrel). Whether the endcap has its own phi windows is open;
                the offset table below decides. Use --phi-mask-endcap to apply
                the mask to the endcap too.

Plots (counts on top, fast-sim fraction with binomial errors below,
       dashed reference line at 0.75 in the fraction panel)
  01_acceptance_2d     theta vs phi fast-sim fraction, 5x5 deg, cells < 3 photons grey
  2a_theta_full        theta, phi mask
  2b_theta_barrel      theta in 40-140, phi mask
  2c_theta_endcap      theta <= 33 or >= 147, phi mask (off by default)
  3a_phi_full          phi, theta gaps removed
  3b_phi_barrel        phi, theta in 40-140
  3c_phi_endcap        phi, endcap core
  4a_energy_barrel     fraction vs E (log axis), barrel, phi mask
  4b_energy_endcap     fraction vs E (log axis), endcap core, phi mask (off by default)

Binomial error sqrt(f(1-f)/N) is zero when f = 0 or 1; read those points with care.

Printed checks
  1. fast-sim rate inside vs outside the phi dead zone (barrel and endcap),
     inside the theta gaps, and in the beam region.
  2. Offset table: fast-sim fraction vs d = phi - C (offset from the nearest of
     the 8 crack centres, all 8 combined), 1 deg bins, barrel and endcap core
     side by side, no phi mask. Barrel expectation: about 0% for d in [0, 8],
     about 90% elsewhere. If the endcap shows the same dip, the same windows apply
     there. The first and last rows (d = -23..-22 and 22..23) are only half
     populated, because |d| <= 22.5.

Output: ~/thesis-ml-sim/plots/[GROUP/]cc3_acceptance_v4_<tag>/

Run
  source ~/source.sh
  python3 cc3_acceptance_v4.py -n 1000
  python3 cc3_acceptance_v4.py
  Options: --group NAME, --phi-mask-endcap
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from podio import root_io
import edm4hep

CC_FILE = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-100kevents-sim.edm4hep.root"
PLOTS   = os.path.expanduser("~/thesis-ml-sim/plots")

PHOTON_PDG  = 22
E_THRESHOLD = 10.0
MIN_ENTRIES_2D = 3
FRACTION_REF = 0.75

THETA_BARREL_LO, THETA_BARREL_HI = 40.0, 140.0
THETA_FWD_MAX, THETA_BWD_MIN     = 33.0, 147.0
ENDCAP_CORE_LO, ENDCAP_CORE_HI   = 12.5, 167.5
TRIGGER_THETA_EDGES = [33.0, 40.0, 140.0, 147.0]

PHI_CRACK_CENTRES = [-157.5 + 45.0 * k for k in range(8)]
PHI_DEAD_LO, PHI_DEAD_HI = -0.01, 8.0
PHI_MARGIN = 0.5

THETA_BINS_1D = np.arange(0.0, 180.0 + 1e-6, 2.5)
PHI_BINS_1D   = np.arange(-180.0, 180.0 + 1e-6, 2.5)
E_BINS        = np.geomspace(10.0, 130.0, 21)
THETA_BINS_2D = np.arange(0.0, 180.0 + 1e-6, 5.0)
PHI_BINS_2D   = np.arange(-180.0, 180.0 + 1e-6, 5.0)
OFFSET_EDGES  = np.arange(-23.0, 23.0 + 1e-6, 1.0)

plt.rcParams.update({
    "font.family": "serif", "font.size": 14, "axes.labelsize": 16,
    "axes.titlesize": 16, "legend.fontsize": 12, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "figure.dpi": 150,
})

ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, default=None, help="max events")
ap.add_argument("--group", default="", help="optional group folder inside plots/")
ap.add_argument("--phi-mask-endcap", action="store_true",
                help="apply the phi dead-zone mask to endcap photons too")
args = ap.parse_args()

if not os.path.exists(CC_FILE):
    sys.exit(f"Input file not found: {CC_FILE}")

tag = "100k" if args.n is None else (f"{args.n // 1000}k" if args.n % 1000 == 0 else str(args.n))
OUT = os.path.join(PLOTS, args.group, f"cc3_acceptance_v4_{tag}")
os.makedirs(OUT, exist_ok=True)
print(f"Output folder: {OUT}")


def load_cc3(path, nmax):
    th, ph, en, fs = [], [], [], []
    n_ev = 0
    for event in root_io.Reader(path).get("events"):
        if nmax is not None and n_ev >= nmax:
            break
        n_ev += 1
        for p in event.get("MCParticles"):
            if p.getPDG() != PHOTON_PDG or p.getGeneratorStatus() != 1:
                continue
            if p.getEnergy() < E_THRESHOLD:
                continue
            if any(abs(q.getPDG()) == 11 for q in p.getParents()):
                continue
            p4 = edm4hep.utils.p4(p)
            th.append(np.degrees(float(p4.theta())))
            ph.append(np.degrees(float(p4.phi())))
            en.append(p.getEnergy())
            fs.append(bool(p.isHandledByFastSim()))
        if n_ev % 20000 == 0:
            print(f"  processed {n_ev} events", flush=True)
    th, ph, en, fs = np.array(th), np.array(ph), np.array(en), np.array(fs, dtype=bool)
    print(f"  Events: {n_ev}  |  photons: {len(th)}  |  fast-sim: {int(fs.sum())} ({100 * fs.mean():.1f}%)")
    return th, ph, en, fs


def in_window(phi, a, b):
    out = np.zeros(len(phi), dtype=bool)
    for c in PHI_CRACK_CENTRES:
        out |= ((phi - c - a) % 360.0) <= (b - a)
    return out


def offset_from_crack(phi):
    best = np.full(len(phi), 999.0)
    for c in PHI_CRACK_CENTRES:
        x = ((phi - c + 180.0) % 360.0) - 180.0
        best = np.where(np.abs(x) < np.abs(best), x, best)
    return best


def fraction_1d(values, fs, edges):
    n_tot, _ = np.histogram(values, bins=edges)
    n_fs, _ = np.histogram(values[fs], bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(n_tot > 0, n_fs / np.maximum(n_tot, 1), np.nan)
        err = np.where(n_tot > 0, np.sqrt(frac * (1 - frac) / np.maximum(n_tot, 1)), np.nan)
    return frac, err, n_tot, n_fs


def plot_fraction(edges, values, fs, xlabel, title, note, fname, xlog=False, ylog=False,
                  lines=None, shade=None, shade_label=None):
    frac, err, n_tot, n_fs = fraction_1d(values, fs, edges)
    lefts, widths = edges[:-1], np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:]) if xlog else 0.5 * (edges[:-1] + edges[1:])

    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
    fig.subplots_adjust(hspace=0.08)
    ax_t.bar(lefts, n_tot, width=widths, align="edge", color="#9a9a9a", alpha=0.75,
             label=f"All photons  (N={int(n_tot.sum()):,})")
    ax_t.bar(lefts, n_fs, width=widths, align="edge", color="#f4a582", alpha=0.95,
             label=f"Fast-sim handled  (N={int(n_fs.sum()):,})")
    ymax = max(float(n_tot.max()), 1.0)
    if ylog:
        ax_t.set_yscale("log")
        ax_t.set_ylim(0.8, ymax * (ymax / 0.8) ** 0.45)
    else:
        ax_t.set_ylim(0, ymax * 1.45)

    if shade:
        for i, (a, b) in enumerate(shade):
            ax_t.axvspan(a, b, color="#4c78a8", alpha=0.15, lw=0,
                         label=shade_label if i == 0 else None)
            ax_b.axvspan(a, b, color="#4c78a8", alpha=0.15, lw=0)
    for positions, color, ls, label in (lines or []):
        for i, v in enumerate(positions):
            ax_t.axvline(v, color=color, ls=ls, lw=1.3, label=label if i == 0 else None)
            ax_b.axvline(v, color=color, ls=ls, lw=1.3)

    ax_b.axhline(FRACTION_REF, color="gray", ls="--", lw=1.0)
    ax_b.errorbar(centers, frac, yerr=err, fmt="o", color="#d6604d", ms=4, lw=1)
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.set_yticks([0.0, 0.25, 0.5, FRACTION_REF, 1.0])
    ax_b.set_xlim(edges[0], edges[-1])
    ax_t.set_ylabel("Photons / bin")
    ax_b.set_ylabel("Fast-sim fraction")
    ax_b.set_xlabel(xlabel)
    ax_t.set_title(title, pad=24)
    ax_t.text(0.0, 1.015, note, transform=ax_t.transAxes, fontsize=11, color="#555555",
              ha="left", va="bottom")
    if xlog:
        ax_b.set_xscale("log")
        ax_b.set_xticks([10, 20, 30, 50, 100])
        ax_b.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_b.xaxis.set_minor_formatter(mticker.NullFormatter())
    for ax in (ax_t, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_t.legend(loc="upper right", frameon=False)
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)
    ntot, nfs = int(n_tot.sum()), int(n_fs.sum())
    print(f"  saved {fname}: N={ntot}, fast-sim={nfs} ({100 * nfs / max(ntot, 1):.1f}%)")


def plot_2d(theta, phi, fs):
    tot, _, _ = np.histogram2d(theta, phi, bins=[THETA_BINS_2D, PHI_BINS_2D])
    hand, _, _ = np.histogram2d(theta[fs], phi[fs], bins=[THETA_BINS_2D, PHI_BINS_2D])
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(tot > 0, hand / np.maximum(tot, 1), np.nan)
    frac = np.ma.masked_where(tot < MIN_ENTRIES_2D, frac).T
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cc3", ["#ffffff", "#f4a582", "#d6604d", "#8b0000"], N=256)
    cmap.set_bad(color="#dddddd")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    im = ax.pcolormesh(THETA_BINS_2D, PHI_BINS_2D, frac, cmap=cmap, vmin=0, vmax=1, shading="flat")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label("Fast-sim fraction")
    for v in TRIGGER_THETA_EDGES:
        ax.axvline(v, color="#2b6cb0", ls=":", lw=1.2)
    ax.set_xlabel(r"Photon $\theta$ [deg]")
    ax.set_ylabel(r"Photon $\phi$ [deg]")
    ax.set_title("Fast-sim fraction, θ vs φ")
    ax.set_xticks(np.arange(0, 181, 15))
    ax.set_yticks(np.arange(-180, 181, 45))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_acceptance_2d.png"), bbox_inches="tight")
    plt.close(fig)
    print("  saved 01_acceptance_2d.png")


print("\nLoading CC3 sim file...")
theta, phi, energy, fs = load_cc3(CC_FILE, args.n)

barrel = (theta >= THETA_BARREL_LO) & (theta <= THETA_BARREL_HI)
endcap_all = (theta <= THETA_FWD_MAX) | (theta >= THETA_BWD_MIN)
endcap_core = (((theta >= ENDCAP_CORE_LO) & (theta <= THETA_FWD_MAX)) |
               ((theta >= THETA_BWD_MIN) & (theta <= ENDCAP_CORE_HI)))
gap = (((theta > THETA_FWD_MAX) & (theta < THETA_BARREL_LO)) |
       ((theta > THETA_BARREL_HI) & (theta < THETA_BWD_MIN)))

crack = in_window(phi, PHI_DEAD_LO - PHI_MARGIN, PHI_DEAD_HI + PHI_MARGIN)
phi_apply = np.ones(len(theta), dtype=bool) if args.phi_mask_endcap else barrel
phi_ok = ~(crack & phi_apply)
theta_ok = ~gap
dead_core = in_window(phi, 0.5, 7.5)


def rate(m):
    n, k = int(m.sum()), int((m & fs).sum())
    return n, k, (100.0 * k / n if n else float("nan"))


print("\nCheck 1: fast-sim rate")
checks = [
    ("barrel, inside dead-zone core [C+0.5, C+7.5]",  barrel & dead_core),
    ("barrel, outside dead window (+-0.5 margin)",     barrel & ~crack),
    ("endcap core, inside dead-zone core",             endcap_core & dead_core),
    ("endcap core, outside dead window",               endcap_core & ~crack),
    ("theta gaps [34,39] and [141,146]",               ((theta >= 34) & (theta <= 39)) | ((theta >= 141) & (theta <= 146))),
    ("beam region theta < 8 or > 172",                 (theta < 8) | (theta > 172)),
]
for name, m in checks:
    n, k, pct = rate(m)
    print(f"  {name:48s} N={n:7d}  fast-simmed={k:7d}  ({pct:.1f}%)")

print("\nCheck 2: fast-sim fraction vs offset d = phi - C from the nearest crack centre")
print("  (no phi mask; dead zone is d in [0, 8])")
d_off = offset_from_crack(phi)
_, _, nb, kb = fraction_1d(d_off[barrel], fs[barrel], OFFSET_EDGES)
_, _, ne, ke = fraction_1d(d_off[endcap_core], fs[endcap_core], OFFSET_EDGES)
print(f"  {'d bin [deg]':14s} {'barrel N':>9s} {'fast %':>7s}   {'endcap N':>9s} {'fast %':>7s}")
for i in range(len(OFFSET_EDGES) - 1):
    lo, hi = OFFSET_EDGES[i], OFFSET_EDGES[i + 1]
    pb = 100.0 * kb[i] / nb[i] if nb[i] else float("nan")
    pe = 100.0 * ke[i] / ne[i] if ne[i] else float("nan")
    mark = "  <- dead zone" if (lo >= 0 and hi <= 8) else ""
    print(f"  [{lo:+4.0f},{hi:+4.0f})    {int(nb[i]):9d} {pb:7.1f}   {int(ne[i]):9d} {pe:7.1f}{mark}")

theta_lines = [(TRIGGER_THETA_EDGES, "#2b6cb0", ":", "Trigger edges 33/40/140/147°")]

print("\nPlotting...")
plot_2d(theta, phi, fs)

phi_note_barrel = "φ dead zones removed (barrel)"
if args.phi_mask_endcap:
    phi_note_theta = "φ dead zones removed (barrel and endcap)"
    phi_note_endcap = "φ dead zones removed"
else:
    phi_note_theta = "φ dead zones removed (barrel photons only)"
    phi_note_endcap = "no φ cut for endcap photons"
TH, PH = r"Photon $\theta$ [deg]", r"Photon $\phi$ [deg]"

for mask, fname, title, note in [
    (phi_ok,              "2a_theta_full.png",   "Fast-sim fraction vs θ",          phi_note_theta),
    (phi_ok & barrel,     "2b_theta_barrel.png", "Fast-sim fraction vs θ, barrel",  phi_note_barrel),
    (phi_ok & endcap_all, "2c_theta_endcap.png", "Fast-sim fraction vs θ, endcaps", phi_note_endcap),
]:
    plot_fraction(THETA_BINS_1D, theta[mask], fs[mask], TH, title, note, fname, lines=theta_lines)

dead_zones = [(c + PHI_DEAD_LO, c + PHI_DEAD_HI) for c in PHI_CRACK_CENTRES]
phi_kw = dict(shade=dead_zones, shade_label="Trigger dead zone [C, C+8°]")
for mask, fname, title, note in [
    (theta_ok,    "3a_phi_full.png",   "Fast-sim fraction vs φ",          "θ gaps removed (33–40°, 140–147°)"),
    (barrel,      "3b_phi_barrel.png", "Fast-sim fraction vs φ, barrel",  "barrel only, θ 40–140°"),
    (endcap_core, "3c_phi_endcap.png", "Fast-sim fraction vs φ, endcaps", "endcap core, θ 12.5–33° and 147–167.5°"),
]:
    plot_fraction(PHI_BINS_1D, phi[mask], fs[mask], PH, title, note, fname, **phi_kw)

for mask, fname, title, note in [
    (phi_ok & barrel,      "4a_energy_barrel.png", "Fast-sim fraction vs energy, barrel",   "barrel, " + phi_note_barrel),
    (phi_ok & endcap_core, "4b_energy_endcap.png", "Fast-sim fraction vs energy, endcaps", "endcap core, " + phi_note_endcap),
]:
    plot_fraction(E_BINS, energy[mask], fs[mask], "Photon energy [GeV]", title, note, fname,
                  xlog=True, ylog=True)

print(f"\nAll plots saved to: {OUT}")
