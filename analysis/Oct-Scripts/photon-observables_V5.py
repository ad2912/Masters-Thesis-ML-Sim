"""
single_photon_observables_v5.py
================================
Reco-level single-photon benchmarking, G4 vs CC3, inside the fast-sim region.

Selection, truth-first (cuts are on the MC photon; the PFO is looked up afterwards)
  Layer A (signal cuts, MC photon): generatorStatus == 1, PDG == 22, E_mc >= 10 GeV,
          NO parent (any of them) with |PDG| == 11.
  Layer B (geometric cuts, MC photon angles): barrel theta 40-140 deg, and phi
          outside the dead window (C - 0.01 - 0.5, C + 8 + 0.5) deg for
          C = -157.5 + 45 k, k = 0..7 (McKeown thesis eq. 9.1 + 0.5 deg margin,
          same function as the S7 acceptance plots).
  Fast-sim flag (MCParticle.isHandledByFastSim()) = SPLIT, not a cut:
          True-CC3    : CC3 file, A+B, flag True
          G4-fallback : CC3 file, A+B, flag False   (Geant4 inside the CC3 run)
          Geant4 (all selected): G4 file, A+B       (no flag used to select)
  True-CC3 + G4-fallback = every CC3-file photon in the region.
  Caveat: the flag is a simulation outcome, so True-CC3 is a subset (about 90%) of
  the region photons. G4-fallback acts as the control: if it differs from
  Geant4 (all selected), True-CC3 vs Geant4 contains a composition effect.

Reco side
  PFO = PandoraPFOs entry with PDG == 22. Link = RecoMCTruthLink (best match per
  PFO, decoded CLUSTER weight >= 0.5, packed weight: cluster = (raw // 10000)/1000).
  Per truth photon: owned PFOs = photon PFOs whose best link is this photon;
  the highest-energy owned PFO is used (split showers: leading PFO only).
  Photons with no owned PFO are dropped from the observables; the matched
  fraction is printed per group and per energy bin.
  n_split = number of links (ANY PFO type) with cluster weight >= 0.5 to the photon.
  Matching key = (collectionID, index) of the object.

Fake rate (backup)
  Reco-first, PFO-angle region (barrel theta, outside dead window, from the PFO
  momentum), binned by reco energy; G4 file vs CC3 file (both groups together),
  no flag split. Fake = photon PFO with no link >= 0.5 to a status-1 MC photon
  (any energy, any parent).

Plots (E_mc axis is log; energy bins start at 10 GeV)
  fig1  mean E_reco/E_mc (SEM bars) and its RMS (std, approx. error std/sqrt(2N))
  fig2a mean cluster width <sigma_perp> + ratio panel to Geant4
        sigma_perp = sqrt(sum E d^2 / sum E), d = distance of hit from the
        IP-to-centroid axis, all hits of clusters[0].
  fig2c radial profile, longitudinally integrated, normalised to the cluster
        energy (fraction per mm^2), 2.5 mm rings, ratio panels (0.8-1.2)
  fig3  splitting: fraction with n_split >= 2 vs E_mc, and n_split distribution
  fig4  fake rate vs reco energy
Printed (also saved to summary.txt)
  events per file, flagged photons in the G4 file (expected 0), selected truth
  photons per group (sum check), matched fractions, summary table, 99.9% energy
  containment radius, and the ZONE TABLE: energy fractions in core (r < 9 mm),
  mid (9-25 mm) and tail (>= 25 mm), the contribution of each zone to <sigma^2>,
  and the mean R90, for all E_mc and split at 40 GeV.

Run
  source ~/source.sh
  python3 single_photon_observables_v5.py --dataset 1k
  nohup python3 single_photon_observables_v5.py --dataset 100k > v5_100k.log 2>&1 &
  Option: --group NAME (folder inside plots/)
"""
import os
import sys
import math
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from podio import root_io
import edm4hep

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["1k", "10k", "100k"], default="1k")
ap.add_argument("--group", default="")
args = ap.parse_args()

DATASET = args.dataset
_REC = "/data/dust/user/alimuham/thesis/reco/"
_PATHS = {
    "1k":   (_REC + "tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root",
             _REC + "tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root", 1000),
    "10k":  (_REC + "tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root",
             _REC + "tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root", 10000),
    "100k": (_REC + "tau-pi0-geant4-100kevents-reco-test_REC.edm4hep.root",
             _REC + "tau-pi0-caloclouds-100kevents-reco-test_REC.edm4hep.root", 100000),
}
G4_FILE, CC3_FILE, MAX_EVENTS = _PATHS[DATASET]
for _f in (G4_FILE, CC3_FILE):
    if not os.path.exists(_f):
        sys.exit(f"Input file not found: {_f}")

PLOTS = os.path.expanduser("~/thesis-ml-sim/plots")
OUT = os.path.join(PLOTS, args.group, f"single_photon_observables_v5_{DATASET}")
os.makedirs(OUT, exist_ok=True)

PHOTON_PDG, ELECTRON_PDG = 22, 11
E_MC_MIN = 10.0
WEIGHT_CUT = 0.5
MIN_N = 5

THETA_BARREL_LO, THETA_BARREL_HI = 40.0, 140.0
PHI_CRACK_CENTRES = [-157.5 + 45.0 * k for k in range(8)]
PHI_DEAD_LO, PHI_DEAD_HI, PHI_MARGIN = -0.01, 8.0, 0.5
_A, _B = PHI_DEAD_LO - PHI_MARGIN, PHI_DEAD_HI + PHI_MARGIN

E_BINS = np.array([10, 20, 30, 40, 60, 80, 100, 125], dtype=float)
N_E_BINS = len(E_BINS) - 1
E_CENTERS = np.sqrt(E_BINS[:-1] * E_BINS[1:])

RADIAL_BIN_WIDTH_MM, RADIAL_MAX_MM, RADIAL_MIN_HITS = 2.5, 50.0, 20
RADIAL_BINS = np.arange(0, RADIAL_MAX_MM + RADIAL_BIN_WIDTH_MM, RADIAL_BIN_WIDTH_MM)
RADIAL_CENTERS = 0.5 * (RADIAL_BINS[:-1] + RADIAL_BINS[1:])
RADIAL_RING_AREA = np.pi * (RADIAL_BINS[1:] ** 2 - RADIAL_BINS[:-1] ** 2)
N_RINGS = len(RADIAL_CENTERS)
FINE_BINS = np.arange(0.0, 301.0, 1.0)
RATIO_YLIM = (0.8, 1.2)
R_CORE, R_MID = 9.0, 25.0
E_SPLIT_GEV = 40.0

GROUPS = ["G4", "CC3", "FB"]
C_G4, C_CC3, C_FB = "#2166ac", "#d6604d", "#7f7f7f"
STYLE = {
    "G4":  dict(color=C_G4,  marker="o", linestyle="-",  linewidth=2.0, markersize=6, label="Geant4 (all selected)"),
    "CC3": dict(color=C_CC3, marker="s", linestyle="-",  linewidth=2.0, markersize=6, label="True-CC3"),
    "FB":  dict(color=C_FB,  marker="^", linestyle="--", linewidth=2.0, markersize=6, label="G4-fallback"),
}
NOTE = "status 1, E ≥ 10 GeV, no e± parent | barrel θ 40–140°, φ dead zones removed"

plt.rcParams.update({
    "font.family": "serif", "font.size": 14, "axes.labelsize": 15,
    "axes.titlesize": 15, "legend.fontsize": 11, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})

LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── helpers ────────────────────────────────────────────────────────────────
def oid(obj):
    o = obj.getObjectID()
    try:
        return (int(o.collectionID), int(o.index))
    except AttributeError:
        return (0, int(o.index))


def decode_weight(raw_weight):
    raw_int = int(round(raw_weight))
    return (raw_int // 10000) / 1000.0, (raw_int % 10000) / 1000.0


def bin_index(value, bins=E_BINS):
    idx = int(np.searchsorted(bins, value, side="right")) - 1
    return idx if 0 <= idx < len(bins) - 1 else -1


def in_dead_window(phi_deg):
    for c in PHI_CRACK_CENTRES:
        if ((phi_deg - c - _A) % 360.0) <= (_B - _A):
            return True
    return False


def layer_a(mc):
    if mc.getGeneratorStatus() != 1 or mc.getPDG() != PHOTON_PDG:
        return False
    if mc.getEnergy() < E_MC_MIN:
        return False
    return not any(abs(q.getPDG()) == ELECTRON_PDG for q in mc.getParents())


def layer_b(mc):
    p4 = edm4hep.utils.p4(mc)
    theta = math.degrees(float(p4.theta()))
    phi = math.degrees(float(p4.phi()))
    if not (THETA_BARREL_LO <= theta <= THETA_BARREL_HI):
        return False
    return not in_dead_window(phi)


def cluster_shape(cluster):
    hits = cluster.getHits()
    if len(hits) < 2:
        return None
    xs = np.array([h.getPosition().x for h in hits], dtype=float)
    ys = np.array([h.getPosition().y for h in hits], dtype=float)
    zs = np.array([h.getPosition().z for h in hits], dtype=float)
    es = np.array([h.getEnergy() for h in hits], dtype=float)
    e_tot = es.sum()
    if e_tot <= 0:
        return None
    cx, cy, cz = np.dot(es, xs) / e_tot, np.dot(es, ys) / e_tot, np.dot(es, zs) / e_tot
    c_mag = math.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
    if c_mag == 0:
        return None
    ax_, ay_, az_ = cx / c_mag, cy / c_mag, cz / c_mag
    r_dot_a = xs * ax_ + ys * ay_ + zs * az_
    d_perp = np.sqrt(np.maximum(xs ** 2 + ys ** 2 + zs ** 2 - r_dot_a ** 2, 0.0))
    sigma = math.sqrt(np.dot(es, d_perp ** 2) / e_tot)
    return sigma, d_perp, es


def zone_quantities(d, es):
    e_tot = es.sum()
    masks = [d < R_CORE, (d >= R_CORE) & (d < R_MID), d >= R_MID]
    frac = [es[m].sum() / e_tot for m in masks]
    s2 = [np.dot(es[m], d[m] ** 2) / e_tot for m in masks]
    order = np.argsort(d)
    cum = np.cumsum(es[order]) / e_tot
    r90 = d[order][min(int(np.searchsorted(cum, 0.9)), len(d) - 1)]
    return frac, s2, r90


def new_acc():
    return dict(
        n_truth=np.zeros(N_E_BINS, int), n_matched=np.zeros(N_E_BINS, int),
        split=[[] for _ in range(N_E_BINS)], rows=[],
        ring_e=np.zeros((N_E_BINS, N_RINGS)), ring_n=np.zeros((N_E_BINS, N_RINGS), int),
        showers=np.zeros(N_E_BINS, int), etot=np.zeros(N_E_BINS),
        fine_e=np.zeros(len(FINE_BINS) - 1),
    )


# ── main pass ──────────────────────────────────────────────────────────────
def run(filepath, file_label, is_cc3):
    log(f"\n{'=' * 60}\n  {file_label}: {filepath}\n{'=' * 60}")
    accs = {}
    fake_all = np.zeros(N_E_BINS, int)
    fake_fake = np.zeros(N_E_BINS, int)
    n_ev = n_flag_in_g4 = 0

    for event in root_io.Reader(filepath).get("events"):
        if n_ev >= MAX_EVENTS:
            break
        n_ev += 1

        reco_to_mc, mc_cws = {}, defaultdict(list)
        for link in event.get("RecoMCTruthLink"):
            cw, _ = decode_weight(link.getWeight())
            mc = link.getTo()
            mc_cws[oid(mc)].append(cw)
            if cw < WEIGHT_CUT:
                continue
            rid = oid(link.getFrom())
            if rid not in reco_to_mc or cw > reco_to_mc[rid][1]:
                reco_to_mc[rid] = (mc, cw)

        mc_to_owned = defaultdict(list)
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue
            rid = oid(pfo)
            if rid in reco_to_mc:
                mc_to_owned[oid(reco_to_mc[rid][0])].append(pfo)
            mom = pfo.getMomentum()
            pmag = math.sqrt(mom.x ** 2 + mom.y ** 2 + mom.z ** 2)
            if pmag == 0:
                continue
            th = math.degrees(math.acos(max(-1.0, min(1.0, mom.z / pmag))))
            ph = math.degrees(math.atan2(mom.y, mom.x))
            if not (THETA_BARREL_LO <= th <= THETA_BARREL_HI) or in_dead_window(ph):
                continue
            fb = bin_index(pfo.getEnergy())
            if fb < 0:
                continue
            fake_all[fb] += 1
            genuine = False
            if rid in reco_to_mc:
                m = reco_to_mc[rid][0]
                genuine = (m.getGeneratorStatus() == 1 and m.getPDG() == PHOTON_PDG)
            if not genuine:
                fake_fake[fb] += 1

        for mc in event.get("MCParticles"):
            if not layer_a(mc) or not layer_b(mc):
                continue
            e_mc = mc.getEnergy()
            b = bin_index(e_mc)
            if b < 0:
                continue
            try:
                flag = bool(mc.isHandledByFastSim())
            except AttributeError:
                sys.exit("isHandledByFastSim() not found. Run 'source ~/source.sh' (Key4hep nightlies).")
            if is_cc3:
                g = "CC3" if flag else "FB"
            else:
                g = "G4"
                n_flag_in_g4 += int(flag)
            acc = accs.setdefault(g, new_acc())
            acc["n_truth"][b] += 1

            mid = oid(mc)
            n_split = sum(1 for w in mc_cws.get(mid, []) if w >= WEIGHT_CUT)
            if n_split >= 1:
                acc["split"][b].append(n_split)

            owned = mc_to_owned.get(mid, [])
            if not owned:
                continue
            acc["n_matched"][b] += 1
            best = max(owned, key=lambda p: p.getEnergy())
            resp = best.getEnergy() / e_mc
            sigma, r90 = float("nan"), float("nan")
            fr, s2 = [float("nan")] * 3, [float("nan")] * 3
            clusters = best.getClusters()
            if len(clusters) > 0:
                shape = cluster_shape(clusters[0])
                if shape is not None:
                    sigma, d, es = shape
                    fr, s2, r90 = zone_quantities(d, es)
                    acc["ring_e"][b] += np.histogram(d, bins=RADIAL_BINS, weights=es)[0]
                    acc["ring_n"][b] += np.histogram(d, bins=RADIAL_BINS)[0]
                    acc["showers"][b] += 1
                    acc["etot"][b] += es.sum()
                    acc["fine_e"] += np.histogram(d, bins=FINE_BINS, weights=es)[0]
            acc["rows"].append((b, e_mc, resp, sigma, *fr, *s2, r90))

        if n_ev % 10000 == 0:
            log(f"  processed {n_ev} events")

    log(f"  Events processed: {n_ev}")
    if not is_cc3:
        log(f"  Photons flagged fast-sim in the G4 file (expected 0): {n_flag_in_g4}")
    for g, acc in accs.items():
        fine = acc["fine_e"]
        if fine.sum() > 0:
            r999 = FINE_BINS[min(int(np.searchsorted(np.cumsum(fine) / fine.sum(), 0.999)) + 1, len(FINE_BINS) - 1)]
            log(f"  [{STYLE[g]['label']}] 99.9% energy-containment radius: {r999:.0f} mm (RADIAL_MAX_MM = {RADIAL_MAX_MM:.0f})")
    return dict(accs=accs, n_events=n_ev, fake_all=fake_all, fake_fake=fake_fake)


# ── statistics ─────────────────────────────────────────────────────────────
def rows_array(acc):
    return np.array(acc["rows"], dtype=float) if acc["rows"] else np.zeros((0, 11))


def binned_stats(rows, col):
    mean = np.full(N_E_BINS, np.nan)
    std = np.full(N_E_BINS, np.nan)
    n = np.zeros(N_E_BINS, int)
    for b in range(N_E_BINS):
        v = rows[rows[:, 0] == b, col]
        v = v[np.isfinite(v)]
        n[b] = len(v)
        if len(v) >= MIN_N:
            mean[b], std[b] = v.mean(), v.std()
    return mean, std, n


def ratio_with_error(num, num_err, den, den_err):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
        err = np.abs(ratio) * np.sqrt((num_err / num) ** 2 + (den_err / den) ** 2)
    return ratio, err


def radial_density(acc):
    out = np.full((N_E_BINS, N_RINGS), np.nan)
    for b in range(N_E_BINS):
        if acc["showers"][b] < MIN_N or acc["etot"][b] <= 0:
            continue
        dens = acc["ring_e"][b] / acc["etot"][b] / RADIAL_RING_AREA
        dens[acc["ring_n"][b] < RADIAL_MIN_HITS] = np.nan
        out[b] = dens
    return out


# ── plotting helpers ───────────────────────────────────────────────────────
def errplot(ax, x, y, yerr, style):
    m = np.isfinite(y)
    err = np.where(np.isfinite(yerr[m]), yerr[m], 0.0)
    ax.errorbar(x[m], y[m], yerr=err, capsize=3, **style)


def energy_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(E_BINS[0], E_BINS[-1])
    ax.set_xticks([10, 20, 30, 50, 100])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def titled(ax, title, note=NOTE):
    ax.set_title(title, pad=24)
    ax.text(0.0, 1.015, note, transform=ax.transAxes, fontsize=10, color="#555555",
            ha="left", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {name}")


def present(R):
    return [g for g in GROUPS if g in R]


# ── figures ────────────────────────────────────────────────────────────────
def fig_response(R):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    fig.subplots_adjust(hspace=0.08)
    for g in present(R):
        mean, std, n = R[g]["resp"]
        with np.errstate(divide="ignore", invalid="ignore"):
            errplot(a1, E_CENTERS, mean, std / np.sqrt(np.maximum(n, 1)), STYLE[g])
            errplot(a2, E_CENTERS, std, std / np.sqrt(2 * np.maximum(n, 1)), STYLE[g])
    a1.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.6)
    a1.set_ylabel(r"Mean $E_\mathrm{reco}/E_\mathrm{mc}$")
    a1.legend(frameon=False, loc="best")
    a1.tick_params(labelbottom=False)
    titled(a1, "Photon energy response")
    a2.set_ylabel(r"RMS of $E_\mathrm{reco}/E_\mathrm{mc}$")
    a2.set_xlabel(r"MC photon energy $E_\mathrm{mc}$ [GeV]")
    a2.set_ylim(bottom=0)
    energy_axis(a2)
    save(fig, "fig1_energy_response.png")


def fig_width(R):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.4]})
    fig.subplots_adjust(hspace=0.08)
    for g in present(R):
        mean, std, n = R[g]["width"]
        with np.errstate(divide="ignore", invalid="ignore"):
            errplot(a1, E_CENTERS, mean, std / np.sqrt(np.maximum(n, 1)), STYLE[g])
    a1.set_ylabel(r"Mean cluster width $\langle\sigma_\perp\rangle$ [mm]")
    a1.legend(frameon=False, loc="best")
    a1.tick_params(labelbottom=False)
    titled(a1, "ECAL cluster transverse width")
    a2.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.6)
    g4m, g4s, g4n = R["G4"]["width"]
    g4e = g4s / np.sqrt(np.maximum(g4n, 1))
    for g in ("CC3", "FB"):
        if g not in R:
            continue
        m, s, n = R[g]["width"]
        ratio, err = ratio_with_error(m, s / np.sqrt(np.maximum(n, 1)), g4m, g4e)
        st = dict(STYLE[g])
        st["label"] = f"{STYLE[g]['label']} / Geant4"
        errplot(a2, E_CENTERS, ratio, err, st)
    a2.set_ylabel("ratio to G4")
    a2.set_xlabel(r"MC photon energy $E_\mathrm{mc}$ [GeV]")
    a2.legend(frameon=False, fontsize=9, loc="best")
    energy_axis(a2)
    save(fig, "fig2a_cluster_width.png")


def fig_radial(R):
    dens = {g: radial_density(R[g]["acc"]) for g in present(R)}
    ncols = 4
    nrows = int(np.ceil(N_E_BINS / ncols))
    fig = plt.figure(figsize=(4.4 * ncols, 5.6 * nrows))
    gs = fig.add_gridspec(nrows * 2, ncols, height_ratios=[3, 1.4] * nrows, hspace=0.1, wspace=0.4)
    first = True
    for b in range(N_E_BINS):
        r, c = divmod(b, ncols)
        ax_m = fig.add_subplot(gs[2 * r, c])
        ax_r = fig.add_subplot(gs[2 * r + 1, c], sharex=ax_m)
        if "G4" not in dens or np.all(np.isnan(dens["G4"][b])):
            ax_m.axis("off")
            ax_r.axis("off")
            continue
        for g in present(R):
            y = np.where(dens[g][b] > 0, dens[g][b], np.nan)
            ax_m.step(RADIAL_CENTERS, y, where="mid", color=STYLE[g]["color"],
                      linestyle=STYLE[g]["linestyle"], label=STYLE[g]["label"])
        ax_m.set_yscale("log")
        ax_m.set_title(f"{E_BINS[b]:.0f}–{E_BINS[b + 1]:.0f} GeV", fontsize=12)
        ax_m.tick_params(labelbottom=False)
        if c == 0:
            ax_m.set_ylabel("energy fraction / mm$^2$", fontsize=11)
        if first:
            ax_m.legend(fontsize=8, frameon=False)
            first = False
        ax_r.axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.6)
        for g in ("CC3", "FB"):
            if g not in dens:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = dens[g][b] / dens["G4"][b]
            ax_r.step(RADIAL_CENTERS, ratio, where="mid", color=STYLE[g]["color"],
                      linestyle=STYLE[g]["linestyle"])
        ax_r.set_ylim(*RATIO_YLIM)
        ax_r.set_xlabel("distance from axis [mm]", fontsize=10)
        if c == 0:
            ax_r.set_ylabel("ratio to G4", fontsize=10)
    fig.suptitle("Radial energy profile, normalised to cluster energy\n" + NOTE, fontsize=12, y=0.99)
    save(fig, "fig2c_radial_profile.png")


def fig_splitting(R):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 2]})
    fig.subplots_adjust(hspace=0.35)
    for g in present(R):
        acc = R[g]["acc"]
        frac = np.full(N_E_BINS, np.nan)
        err = np.full(N_E_BINS, np.nan)
        for b in range(N_E_BINS):
            v = np.array(acc["split"][b], dtype=int)
            if len(v) >= MIN_N:
                p = float((v >= 2).mean())
                frac[b], err[b] = p, math.sqrt(p * (1 - p) / len(v))
        errplot(a1, E_CENTERS, frac, err, STYLE[g])
    a1.set_ylabel(r"Fraction with $n_\mathrm{split}\geq 2$")
    a1.set_xlabel(r"MC photon energy $E_\mathrm{mc}$ [GeV]")
    a1.set_ylim(bottom=0)
    a1.legend(frameon=False)
    energy_axis(a1)
    titled(a1, "Shower splitting: PFOs per truth photon")
    cats = ["2", "3", "4", "5+"]
    x = np.arange(len(cats))
    width = 0.26
    for i, g in enumerate(present(R)):
        acc = R[g]["acc"]
        vals = np.concatenate([np.array(acc["split"][b], dtype=int) for b in range(N_E_BINS)])
        sv = vals[vals >= 2]
        if len(sv) == 0:
            continue
        counts = np.array([(sv == 2).sum(), (sv == 3).sum(), (sv == 4).sum(), (sv >= 5).sum()]) / len(sv)
        a2.bar(x + (i - 1) * width, counts, width, color=STYLE[g]["color"], alpha=0.85,
               label=STYLE[g]["label"])
    a2.set_xticks(x)
    a2.set_xticklabels(cats)
    a2.set_xlabel(r"$n_\mathrm{split}$ (split photons only, all energies)")
    a2.set_ylabel("fraction of split photons")
    a2.legend(frameon=False, fontsize=9)
    save(fig, "fig3_splitting.png")


def fig_fake(res_g4, res_cc3):
    fig, ax = plt.subplots(figsize=(9, 6))
    for res, key in ((res_g4, "G4"), (res_cc3, "CC3")):
        n_all = res["fake_all"].astype(float)
        n_fake = res["fake_fake"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(n_all > 0, n_fake / n_all, np.nan)
            err = np.sqrt(rate * (1 - rate) / n_all)
        st = dict(STYLE[key])
        st["label"] = "Geant4 file" if key == "G4" else "CC3 file (both groups)"
        errplot(ax, E_CENTERS, rate, err, st)
    ax.set_xlabel(r"Reconstructed photon PFO energy $E_\mathrm{reco}$ [GeV]")
    ax.set_ylabel("Fake rate")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    energy_axis(ax)
    titled(ax, "Photon fake rate", "photon PFOs in the barrel region (PFO angles); fake = no link ≥ 0.5 to a status-1 MC photon")
    save(fig, "fig4_fake_rate.png")


# ── printed summaries ──────────────────────────────────────────────────────
def print_summary(R):
    log("\n=== Selected truth photons (Layer A+B) ===")
    n = {g: int(R[g]["acc"]["n_truth"].sum()) for g in present(R)}
    for g in present(R):
        log(f"  {STYLE[g]['label']:24s} {n[g]:9d}")
    if "G4" in n and "CC3" in n:
        n_cc = n["CC3"] + n.get("FB", 0)
        log(f"  CC3 file (True-CC3 + fallback) = {n_cc}; Geant4 file = {n['G4']}; "
            f"difference = {n_cc - n['G4']} ({100.0 * (n_cc - n['G4']) / max(n['G4'], 1):+.2f}%)")
        log(f"  Fast-sim fraction of the selected CC3-file photons: {100.0 * n['CC3'] / max(n_cc, 1):.1f}%")

    log("\n=== Matched fraction (truth photon has a photon PFO), per E_mc bin ===")
    log("  " + "bin [GeV]".ljust(14) + "".join(STYLE[g]["label"][:18].ljust(22) for g in present(R)))
    for b in range(N_E_BINS):
        line = f"  [{E_BINS[b]:.0f},{E_BINS[b + 1]:.0f})".ljust(16)
        for g in present(R):
            a = R[g]["acc"]
            t, m = a["n_truth"][b], a["n_matched"][b]
            line += (f"{100.0 * m / t:5.1f}% (N={t})" if t else "-").ljust(22)
        log(line)
    for g in present(R):
        a = R[g]["acc"]
        log(f"  overall {STYLE[g]['label']}: {100.0 * a['n_matched'].sum() / max(a['n_truth'].sum(), 1):.1f}%")

    log("\n=== Summary table (all E_mc, matched photons) ===")
    log("  " + "group".ljust(24) + "mean resp".ljust(12) + "RMS resp".ljust(12) + "mean width [mm]".ljust(18) + "split frac")
    for g in present(R):
        a = R[g]["acc"]
        rows = rows_array(a)
        sp = np.concatenate([np.array(a["split"][b], dtype=int) for b in range(N_E_BINS)])
        sw = rows[np.isfinite(rows[:, 3]), 3]
        log("  " + STYLE[g]["label"].ljust(24)
            + f"{rows[:, 2].mean():.4f}".ljust(12) + f"{rows[:, 2].std():.4f}".ljust(12)
            + f"{sw.mean():.3f}".ljust(18) + f"{(sp >= 2).mean():.4f}")


def print_zone_table(R):
    log(f"\n=== ZONE TABLE: core r < {R_CORE:.0f} mm, mid {R_CORE:.0f}-{R_MID:.0f} mm, tail >= {R_MID:.0f} mm ===")
    log("  fractions = mean over photons of the cluster-energy fraction in the zone")
    log("  s2 = mean over photons of sum_zone(E d^2)/E_tot [mm^2]; the three s2 add up to <sigma^2>")
    sel = {"all E_mc": lambda e: np.ones(len(e), bool),
           f"E_mc < {E_SPLIT_GEV:.0f} GeV": lambda e: e < E_SPLIT_GEV,
           f"E_mc >= {E_SPLIT_GEV:.0f} GeV": lambda e: e >= E_SPLIT_GEV}
    for sname, f in sel.items():
        log(f"\n  --- {sname} ---")
        log("  " + "group".ljust(24) + "N".ljust(8) + "frac c/m/t [%]".ljust(24) + "s2 c/m/t [mm^2]".ljust(26) + "<s2>".ljust(9) + "<R90>[mm]")
        ref = None
        out = {}
        for g in present(R):
            rows = rows_array(R[g]["acc"])
            rows = rows[np.isfinite(rows[:, 3])]
            rows = rows[f(rows[:, 1])]
            if len(rows) < MIN_N:
                log("  " + STYLE[g]["label"].ljust(24) + f"{len(rows)} (too few)")
                continue
            fr = rows[:, 4:7].mean(axis=0)
            s2 = rows[:, 7:10].mean(axis=0)
            out[g] = s2
            log("  " + STYLE[g]["label"].ljust(24) + str(len(rows)).ljust(8)
                + (f"{100 * fr[0]:.1f}/{100 * fr[1]:.1f}/{100 * fr[2]:.1f}").ljust(24)
                + (f"{s2[0]:.2f}/{s2[1]:.2f}/{s2[2]:.2f}").ljust(26)
                + f"{s2.sum():.2f}".ljust(9) + f"{rows[:, 10].mean():.2f}")
        if "G4" in out:
            for g in ("CC3", "FB"):
                if g in out:
                    d = out[g] - out["G4"]
                    log(f"  delta <s2> {STYLE[g]['label']} - Geant4 (c/m/t) [mm^2]: "
                        f"{d[0]:+.2f}/{d[1]:+.2f}/{d[2]:+.2f}  total {d.sum():+.2f}")


# ── entry point ────────────────────────────────────────────────────────────
log("=" * 60)
log(f"  single_photon_observables_v5.py   dataset: {DATASET}")
log(f"  Output folder: {OUT}")
log("=" * 60)

res_g4 = run(G4_FILE, "Geant4 reco file", is_cc3=False)
res_cc3 = run(CC3_FILE, "CC3 reco file", is_cc3=True)

R = {}
for res in (res_g4, res_cc3):
    for g, acc in res["accs"].items():
        rows = rows_array(acc)
        R[g] = dict(acc=acc, resp=binned_stats(rows, 2), width=binned_stats(rows, 3))

log("\nGenerating figures...")
fig_response(R)
fig_width(R)
fig_radial(R)
fig_splitting(R)
fig_fake(res_g4, res_cc3)

print_summary(R)
print_zone_table(R)

with open(os.path.join(OUT, "summary.txt"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
log(f"\n  Figures and summary.txt saved to: {OUT}\n  Done.")
