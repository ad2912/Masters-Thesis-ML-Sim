"""
pi0_invariant_mass_v6_3.py
==========================
pi0 -> gamma gamma invariant-mass analysis (G4 vs CC3), MC-first selection.

COLLECTIONS USED (what the experts will ask)
  MCParticles      truth: status-1 photons whose parent is a pi0 (PDG 111); truth E, momentum,
                   isHandledByFastSim() flag (only meaningful in the CC3 file)
  RecoMCTruthLink  reco->MC links; weight packs cluster and track weights;
                   cluster weight = (raw // 10000)/1000; used for photon matching
  PandoraPFOs      reconstructed particle-flow objects; only PDG == 22 (photon PFOs) are used;
                   their 4-momentum gives the invariant mass

SELECTION (state this on the slide)
  MC-first. Start from MC pi0s that have exactly two status-1 photon daughters. Each daughter is
  matched to photon PFOs via RecoMCTruthLink (cluster weight >= 0.5). A pi0 gives a mass only if
  it has EXACTLY two matched photon PFOs that belong to the two DIFFERENT daughter photons.
  No combinatorial pairing. pi0s with 0, 1 (merged/lost) or 3+ (split) matched PFOs, or two PFOs
  from the same daughter (split photon + lost partner), are excluded and counted in the table.

FOUR GATES (defined on the TRUTH daughters, both photons must pass)
  all          : every MC pi0 with exactly two photon daughters, no cut
  e10          : both E_mc >= 10 GeV
  region       : e10 + barrel theta 40-140 deg + outside the phi dead window
                 (C - 0.01 - 0.5, C + 8 + 0.5) deg, C = -157.5 + 45k   (same as single-photon Layer B)
  region_true  : CC3 file: region AND both daughters isHandledByFastSim()==True ("True-CC3" pi0s)
                 G4 file : region (no flag exists/needed)

Changes from v6.2 (v6.3):
  - Theta 37.5-142.5 -> 40-140 (thesis trigger edges, same as single-photon script).
  - Phi window now includes the 0.5 deg safety margin (same function as single-photon script).
  - Link cut is cluster weight >= 0.5 (was > 0.5).
  - Object keys are (collectionID, index).
  - 'all' gate now counts only pi0s with exactly two photon daughters (Dalitz etc. removed).
  - Two PFOs from the SAME daughter photon no longer produce a mass (counted as split).
  - Stale '84.4%' reference removed from the region check printout.
  - Energy profile bins start at 20 GeV (both photons >= 10 GeV) and extend to 250.

Run on NAF:
    source ~/source.sh
    python3 pi0_invariant_mass_v6_3.py
"""

import os, math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

try:
    from podio import root_io
except ImportError:
    root_io = None

# ============================================================
# CONFIG
# ============================================================
RUN_SIZE  = "1k"             # "1k" | "10k" | "100k"
PARALLEL  = True
USE_CACHE = True

RUN_TABLE = {
    "1k": dict(
        g4  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root",
        cc3 = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root",
        max_events = 1000, label = "1k events [dev]"),
    "10k": dict(
        g4  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root",
        cc3 = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root",
        max_events = 10000, label = "10k events"),
    "100k": dict(
        g4  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-geant4-100kevents-reco-test_REC.edm4hep.root",
        cc3 = "/data/dust/user/alimuham/thesis/reco/tau-pi0-caloclouds-100kevents-reco-test_REC.edm4hep.root",
        max_events = 100000, label = "100k events"),
}
import sys
if len(sys.argv) > 1 and sys.argv[1] in RUN_TABLE:
    RUN_SIZE = sys.argv[1]
_cfg = RUN_TABLE[RUN_SIZE]
G4_FILE, CC3_FILE = _cfg["g4"], _cfg["cc3"]
MAX_EVENTS, RUN_LABEL = _cfg["max_events"], _cfg["label"]

PLOT_DIR = os.environ.get(
    "PI0_PLOT_DIR",
    os.path.expanduser(f"~/thesis-ml-sim/plots/pi0_invariant_mass_{RUN_SIZE}_v6_3"))
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG, PI0_PDG = 22, 111
WEIGHT_CUT   = 0.5
E_MC_TIER2   = 10.0
PI0_MASS_MEV = 134.97
MU0          = PI0_MASS_MEV / 1000.0

# ---- Layer B: same as single_photon_observables_v5.py ----
REGION_E_MIN     = 10.0
BARREL_THETA_DEG = (40.0, 140.0)
PHI_CRACK_CENTERS_DEG = [-157.5 + 45.0 * k for k in range(8)]
PHI_DEAD_LO, PHI_DEAD_HI, PHI_MARGIN = -0.01, 8.0, 0.5
_PA, _PB = PHI_DEAD_LO - PHI_MARGIN, PHI_DEAD_HI + PHI_MARGIN

GATES      = ["all", "e10", "region", "region_true"]
GATE_TAG   = {"all": "", "e10": "_E10", "region": "_region", "region_true": "_regionTrue"}
GATE_TITLE = {
    "all":         "All matched pi0s\n(no energy / angular cut)",
    "e10":         "Both photons E_mc >= 10 GeV\n(no angular cut)",
    "region":      "Both photons E_mc >= 10 GeV + barrel,\noutside phi cracks (geometry only)",
    "region_true": "Same geometry; CC3: both photons\nfast-sim-flagged (True-CC3 pi0s)",
}
GATE_LONG = {
    "all":         "all pi0s with two photon daughters",
    "e10":         "both photons E_mc >= 10 GeV",
    "region":      "both photons E_mc >= 10 GeV, barrel 40-140, outside phi cracks (geometry only)",
    "region_true": "same geometry; CC3: both photons flagged fast-sim (True-CC3); G4: geometry",
}
SELECTION_FOOTER = (
    "Selection (MC-first): MC pi0 -> its two truth photons -> matched photon PFOs "
    "(RecoMCTruthLink, cluster weight >= 0.5).\n"
    "Only pi0s with exactly two matched PFOs (one per daughter) enter; no combinatorial pairing. "
    "Gates are defined on the truth photons, identically for G4 and CC3."
)

CB_TAIL_SIDE = "high"
FIT_WINDOWS  = {"gauss": (0.110, 0.170), "cb": (0.100, 0.180)}
MODEL_NAME   = {"gauss": "Gaussian", "cb": "Crystal Ball"}

TRUTH_ZOOM_LO_MEV, TRUTH_ZOOM_HI_MEV, TRUTH_ZOOM_BIN_MEV = 134.5, 135.5, 0.1

M_BINS           = np.arange(0.080, 0.3201, 0.002)
E_RESPONSE_BINS  = np.arange(0.80, 1.21, 0.01)
ANGLE_RATIO_BINS = np.arange(0.70, 1.31, 0.01)
EPI0_BINS        = [20, 30, 40, 60, 80, 125, 250]

C_G4, C_CC3 = "#2166ac", "#d6604d"

plt.rcParams.update({
    "font.family": "serif", "font.size": 14, "axes.titlesize": 15,
    "axes.labelsize": 15, "legend.fontsize": 12, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

LOG = []
def log(s=""):
    print(s, flush=True)
    LOG.append(s)

# ============================================================
# HELPERS
# ============================================================
def oid(obj):
    o = obj.getObjectID()
    try:
        return (int(o.collectionID), int(o.index))
    except AttributeError:
        return (0, int(o.index))


def m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2):
    return math.sqrt(max(2.0 * (e1*e2 - px1*px2 - py1*py2 - pz1*pz2), 0.0))


def decode_weight(raw_weight):
    raw_int = int(round(raw_weight))
    return (raw_int // 10000) / 1000.0, (raw_int % 10000) / 1000.0


def in_fastsim_region(e, px, py, pz):
    if e < REGION_E_MIN:
        return False
    p = math.sqrt(px*px + py*py + pz*pz)
    if p <= 0:
        return False
    theta = math.degrees(math.acos(max(-1.0, min(1.0, pz / p))))
    phi   = math.degrees(math.atan2(py, px))
    if not (BARREL_THETA_DEG[0] <= theta <= BARREL_THETA_DEG[1]):
        return False
    for c in PHI_CRACK_CENTERS_DEG:
        if ((phi - c - _PA) % 360.0) <= (_PB - _PA):
            return False
    return True


def _fastsim_flag(mc):
    try:
        return bool(mc.isHandledByFastSim())
    except Exception:
        return None


def gauss_pure(x, N, mu, sigma):
    return N * np.exp(-0.5 * ((x - mu) / sigma)**2)


def crystal_ball(x, N, mu, sigma, alpha, n):
    z = (x - mu) / sigma
    if CB_TAIL_SIDE == "high":
        z = -z
    a = np.abs(alpha)
    A = (n / a) ** n * np.exp(-0.5 * a**2)
    B = n / a - a
    with np.errstate(invalid="ignore", divide="ignore"):
        return N * np.where(z > -a, np.exp(-0.5 * z**2), A * (B - z) ** (-n))


def _config_signature():
    return repr((RUN_SIZE, MAX_EVENTS, WEIGHT_CUT, E_MC_TIER2, REGION_E_MIN, BARREL_THETA_DEG,
                 PHI_CRACK_CENTERS_DEG, PHI_DEAD_LO, PHI_DEAD_HI, PHI_MARGIN,
                 EPI0_BINS, G4_FILE, CC3_FILE, "v6.3"))

# ============================================================
# ONE PASS OVER A FILE (truth + reco + all gates)
# ============================================================
_ARR_KEYS = ("masses", "lead", "nfrag", "resp", "angle", "epi0_m", "epi0_a")
_EFF_KEYS = ["n_mc_total", "n_0reco", "n_1reco", "n_exactly2", "n_3plus"]
_CHK_KEYS = ["n_e10", "n_region", "n_flag", "n_both", "n_flag_unavail",
             "reg_pi0_0fl", "reg_pi0_1fl", "reg_pi0_2fl", "n_same_photon"]


def _new_store():
    return dict(n_mc=0, n_0=0, n_1=0, n_2=0, n_3=0, masses=[], lead=[], nfrag=[],
                resp=[], angle=[], epi0_m=[], epi0_a=[])


def process_file(filepath, label):
    reader = root_io.Reader(filepath)
    S = {g: _new_store() for g in GATES}
    truth_masses = []
    chk = {k: 0 for k in _CHK_KEYS}
    n_events = 0

    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1
        if n_events % 10000 == 0:
            print(f"  [{label}] {n_events} events", flush=True)

        daughters = defaultdict(list)
        for mc in event.get("MCParticles"):
            if mc.getGeneratorStatus() != 1 or mc.getPDG() != PHOTON_PDG:
                continue
            parents = mc.getParents()
            if len(parents) == 0 or parents[0].getPDG() != PI0_PDG:
                continue
            mom = mc.getMomentum()
            e = mc.getEnergy()
            fs = _fastsim_flag(mc)
            daughters[oid(parents[0])].append((e, mom.x, mom.y, mom.z, fs))
            if e >= E_MC_TIER2:
                reg = in_fastsim_region(e, mom.x, mom.y, mom.z)
                chk["n_e10"] += 1
                chk["n_region"] += int(reg)
                if fs is None:
                    chk["n_flag_unavail"] += 1
                else:
                    chk["n_flag"] += int(fs)
                    chk["n_both"] += int(fs and reg)

        members = {g: set() for g in GATES}
        for pid, d in daughters.items():
            if len(d) == 2:
                members["all"].add(pid)
                truth_masses.append(m_gg(*d[0][:4], *d[1][:4]))
                if min(x[0] for x in d) >= E_MC_TIER2:
                    members["e10"].add(pid)
                    if all(in_fastsim_region(*x[:4]) for x in d):
                        members["region"].add(pid)
                        nfl = sum(1 for x in d if x[4])
                        chk[f"reg_pi0_{nfl}fl"] += 1
                        if label != "CC3" or nfl == 2:
                            members["region_true"].add(pid)
        for g in GATES:
            S[g]["n_mc"] += len(members[g])

        reco_to_mc, reco_to_wgt = {}, {}
        for link in event.get("RecoMCTruthLink"):
            cw, _ = decode_weight(link.getWeight())
            if cw < WEIGHT_CUT:
                continue
            rid = oid(link.getFrom())
            if rid not in reco_to_wgt or cw > reco_to_wgt[rid]:
                reco_to_mc[rid], reco_to_wgt[rid] = link.getTo(), cw

        pi0_reco = defaultdict(list)
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue
            rid = oid(pfo)
            if rid not in reco_to_mc:
                continue
            mc = reco_to_mc[rid]
            if mc.getGeneratorStatus() != 1 or mc.getPDG() != PHOTON_PDG:
                continue
            mp_ = mc.getParents()
            if len(mp_) == 0 or mp_[0].getPDG() != PI0_PDG:
                continue
            mom, mm = pfo.getMomentum(), mc.getMomentum()
            pi0_reco[oid(mp_[0])].append(
                (pfo.getEnergy(), mom.x, mom.y, mom.z, mc.getEnergy(), mm.x, mm.y, mm.z, oid(mc)))

        seen = set()
        for pid, ph in pi0_reco.items():
            if pid not in members["all"]:
                continue
            seen.add(pid)
            gs = [g for g in GATES if pid in members[g]]
            n = len(ph)
            if n == 2 and ph[0][8] == ph[1][8]:
                chk["n_same_photon"] += 1
                for g in gs: S[g]["n_3"] += 1
                en = sorted([p[0] for p in ph], reverse=True)
                for g in gs:
                    S[g]["lead"].append(en[0] / sum(en))
                    S[g]["nfrag"].append(2)
                continue
            if n == 1:
                for g in gs: S[g]["n_1"] += 1
            elif n == 2:
                for g in gs: S[g]["n_2"] += 1
                (e1, px1, py1, pz1, emc1, mx1, my1, mz1, _) = ph[0]
                (e2, px2, py2, pz2, emc2, mx2, my2, mz2, _) = ph[1]
                mass = m_gg(e1, px1, py1, pz1, e2, px2, py2, pz2)
                epi0 = emc1 + emc2
                resp = [e1 / emc1 if emc1 > 0 else None, e2 / emc2 if emc2 > 0 else None]
                p1, p2 = math.sqrt(px1**2+py1**2+pz1**2), math.sqrt(px2**2+py2**2+pz2**2)
                q1, q2 = math.sqrt(mx1**2+my1**2+mz1**2), math.sqrt(mx2**2+my2**2+mz2**2)
                ratio = None
                if min(p1, p2, q1, q2) > 0:
                    c_r = (px1*px2 + py1*py2 + pz1*pz2) / (p1 * p2)
                    c_m = (mx1*mx2 + my1*my2 + mz1*mz2) / (q1 * q2)
                    a_r = math.acos(max(-1.0, min(1.0, c_r)))
                    a_m = math.acos(max(-1.0, min(1.0, c_m)))
                    if a_m > 0:
                        ratio = a_r / a_m
                for g in gs:
                    S[g]["masses"].append(mass)
                    S[g]["epi0_m"].append(epi0)
                    S[g]["resp"].extend([r for r in resp if r is not None])
                    if ratio is not None:
                        S[g]["angle"].append(ratio)
                        S[g]["epi0_a"].append(epi0)
            elif n >= 3:
                for g in gs: S[g]["n_3"] += 1
                en = sorted([p[0] for p in ph], reverse=True)
                tot = sum(en)
                if tot > 0:
                    for g in gs:
                        S[g]["lead"].append(en[0] / tot)
                        S[g]["nfrag"].append(n)
        for g in GATES:
            S[g]["n_0"] += len(members[g] - seen)

    res = dict(label=label, n_events=n_events, truth=np.array(truth_masses), chk=chk, gates={})
    for g in GATES:
        s = S[g]
        res["gates"][g] = dict(
            eff=dict(n_mc_total=s["n_mc"], n_0reco=s["n_0"], n_1reco=s["n_1"],
                     n_exactly2=s["n_2"], n_3plus=s["n_3"],
                     eff_pct=100.0 * s["n_2"] / max(s["n_mc"], 1)),
            **{k: np.array(s[k]) for k in _ARR_KEYS})
    return res

# ============================================================
# CACHE
# ============================================================
def _cache_path(label):
    return f"{PLOT_DIR}/pi0_arrays_{label}.npz"


def save_cache(res):
    d = dict(sig=np.array(_config_signature()), n_events=np.array(res["n_events"]),
             truth=res["truth"], chk=np.array([res["chk"][k] for k in _CHK_KEYS]))
    for g, v in res["gates"].items():
        d[f"{g}__eff"] = np.array([v["eff"][k] for k in _EFF_KEYS])
        for k in _ARR_KEYS:
            d[f"{g}__{k}"] = v[k]
    np.savez(_cache_path(res["label"]), **d)


def load_cache(label):
    p = _cache_path(label)
    if not (USE_CACHE and os.path.exists(p)):
        return None
    z = np.load(p, allow_pickle=False)
    if str(z["sig"]) != _config_signature():
        print(f"  cache for {label} is stale (config changed) -> re-reading ROOT")
        return None
    res = dict(label=label, n_events=int(z["n_events"]), truth=z["truth"],
               chk=dict(zip(_CHK_KEYS, [int(x) for x in z["chk"]])), gates={})
    for g in GATES:
        e = dict(zip(_EFF_KEYS, [int(x) for x in z[f"{g}__eff"]]))
        e["eff_pct"] = 100.0 * e["n_exactly2"] / max(e["n_mc_total"], 1)
        res["gates"][g] = dict(eff=e, **{k: z[f"{g}__{k}"] for k in _ARR_KEYS})
    print(f"  loaded cache: {p}")
    return res

# ============================================================
# FITS
# ============================================================
def _fit(model, centers, counts, label):
    lo, hi = FIT_WINDOWS[model]
    m = (centers > lo) & (centers < hi)
    x, y = centers[m], counts[m].astype(float)
    if len(x) < 8 or y.max() < 5:
        log(f"  {MODEL_NAME[model]} fit [{label}] SKIPPED: insufficient statistics")
        return None, None
    if model == "gauss":
        f, p0 = gauss_pure, [y.max(), MU0, 0.015]
        bounds = ([0, 0.100, 0.003], [1e9, 0.200, 0.040])
    else:
        f, p0 = crystal_ball, [y.max(), MU0, 0.015, 1.5, 3.0]
        bounds = ([0, 0.100, 0.003, 0.3, 1.1], [1e9, 0.200, 0.040, 10.0, 50.0])
    try:
        popt, pcov = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=30000)
    except (RuntimeError, ValueError) as e:
        log(f"  {MODEL_NAME[model]} fit [{label}] FAILED: {e}")
        return None, None
    perr = np.sqrt(np.diag(pcov))
    log(f"  {MODEL_NAME[model]:12s} [{label:3s}] mu={1000*popt[1]:.2f}+/-{1000*perr[1]:.2f} MeV | "
        f"sigma={1000*popt[2]:.2f}+/-{1000*perr[2]:.2f} MeV | bias={1000*popt[1]-PI0_MASS_MEV:+.2f} MeV "
        f"(unweighted LSQ errors)")
    return popt, perr

# ============================================================
# PLOTS: MASS
# ============================================================
def draw_mass_panel(ax, g4_m, cc3_m, model, title, gate_name, logy=False):
    ax.hist(g4_m, bins=M_BINS, histtype="step", color=C_G4, linewidth=2.2,
            label=f"Geant4 (N={len(g4_m)})")
    ax.hist(cc3_m, bins=M_BINS, histtype="step", color=C_CC3, linewidth=2.2, linestyle="--",
            label=f"CC3 (N={len(cc3_m)})")
    ax.axvline(MU0, color="gray", lw=1.0, ls=":", label=f"PDG: {PI0_MASS_MEV:.0f} MeV")
    lo, hi = FIT_WINDOWS[model]
    func = gauss_pure if model == "gauss" else crystal_ball
    texts = [f"{MODEL_NAME[model]} fit, window {1000*lo:.0f}-{1000*hi:.0f} MeV"]
    log(f"\n  --- {gate_name}: {MODEL_NAME[model]} ---")
    for arr, fl, col in ((g4_m, "G4", C_G4), (cc3_m, "CC3", C_CC3)):
        counts, _ = np.histogram(arr, bins=M_BINS)
        centers = 0.5 * (M_BINS[:-1] + M_BINS[1:])
        popt, _ = _fit(model, centers, counts, fl)
        if popt is not None:
            xf = np.linspace(lo, hi, 600)
            ax.plot(xf, func(xf, *popt), color=col, lw=2.0, alpha=0.85)
            texts.append(f"{fl}: mu={1000*popt[1]:.1f}, sigma={1000*popt[2]:.1f}, "
                         f"bias={1000*popt[1]-PI0_MASS_MEV:+.1f} MeV")
    ax.text(0.97, 0.97, "\n".join(texts), transform=ax.transAxes, fontsize=11,
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    ax.set_xlim(M_BINS[0], M_BINS[-1])
    ax.set_xlabel(r"$M_{\gamma\gamma}$ [GeV]")
    ax.set_ylabel("pi0 candidates / 2 MeV")
    ax.set_title(title, fontsize=14, fontweight="bold")
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        ax.legend(framealpha=0.6, loc="lower right")
    else:
        ax.legend(framealpha=0.6, loc="center right", bbox_to_anchor=(1.0, 0.55))


def plot_reco_mass(R4, RC):
    for model in ("gauss", "cb"):
        for logy in ((False, True) if model == "cb" else (False,)):
            suf = "_logy" if logy else ""
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            for ax, g in zip(axes.ravel(), GATES):
                draw_mass_panel(ax, R4["gates"][g]["masses"], RC["gates"][g]["masses"],
                                model, GATE_TITLE[g], g, logy)
            fig.suptitle(f"Reco M(gamma gamma) of matched pi0 daughters: G4 vs CC3  [{RUN_LABEL}]",
                         fontsize=17, y=1.01)
            plt.tight_layout()
            fig.text(0.5, -0.005, SELECTION_FOOTER, ha="center", va="top", fontsize=12)
            name = f"reco_mgg_4gates_{'crystalball' if model == 'cb' else 'gaussian'}{suf}.png"
            fig.savefig(f"{PLOT_DIR}/{name}", bbox_inches="tight")
            plt.close(fig)
            log(f"\n  Saved: {name}")
    for g in ("region", "region_true"):
        for model in ("gauss", "cb"):
            for logy in ((False, True) if model == "cb" else (False,)):
                fig, ax = plt.subplots(figsize=(10, 7.5))
                draw_mass_panel(ax, R4["gates"][g]["masses"], RC["gates"][g]["masses"],
                                model, GATE_TITLE[g], g, logy)
                plt.tight_layout()
                fig.text(0.5, -0.01, SELECTION_FOOTER, ha="center", va="top", fontsize=10)
                name = (f"reco_mgg_{g}_only_{'crystalball' if model == 'cb' else 'gaussian'}"
                        f"{'_logy' if logy else ''}.png")
                fig.savefig(f"{PLOT_DIR}/{name}", bbox_inches="tight")
                plt.close(fig)
                log(f"  Saved: {name}")


def plot_truth_zoomed(t4, tc):
    lo, hi, bw = TRUTH_ZOOM_LO_MEV, TRUTH_ZOOM_HI_MEV, TRUTH_ZOOM_BIN_MEV
    bins = np.linspace(lo, hi, round((hi - lo) / bw) + 1)
    for name, t in (("Truth G4", t4), ("Truth CC3", tc)):
        mm = 1000 * np.mean(t)
        log(f"  {name} mean/std [MeV]: {mm:.4f} / {1000*np.std(t):.4f}")
        if not lo <= mm <= hi:
            log(f"  WARNING: {name} mean outside zoom window [{lo}, {hi}] MeV")
    n4 = int(np.sum((1000*t4 >= lo) & (1000*t4 < hi)))
    nc = int(np.sum((1000*tc >= lo) & (1000*tc < hi)))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(1000*t4, bins=bins, histtype="step", color=C_G4, linewidth=2,
            label=f"Geant4 (N in window={n4}/{len(t4)})")
    ax.hist(1000*tc, bins=bins, histtype="step", color=C_CC3, linewidth=2, linestyle="--",
            label=f"CC3 (N in window={nc}/{len(tc)})")
    ax.axvline(PI0_MASS_MEV, color="black", lw=1.0, ls=":", alpha=0.7, label=f"PDG ({PI0_MASS_MEV} MeV)")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$M_{\gamma\gamma}$ [MeV]")
    ax.set_ylabel(f"pi0 candidates / {bw:.1f} MeV")
    ax.set_title("MC truth M(gamma gamma) of the two pi0 daughter photons\n"
                 f"fixed window {lo}-{hi} MeV, {bw} MeV bins, no cuts", fontsize=13)
    ax.legend(framealpha=0.6)
    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/analysisA_truth_mgg_zoomed.png", bbox_inches="tight")
    plt.close(fig)
    log("  Saved: analysisA_truth_mgg_zoomed.png")

# ============================================================
# PLOTS: BIAS VS PI0 ENERGY
# ============================================================
def _med_err(v):
    h = 0.5 * (np.percentile(v, 84) - np.percentile(v, 16))
    return np.median(v), 1.2533 * h / math.sqrt(len(v))


def _profile(vals, epi0):
    xs, ys, es, ns = [], [], [], []
    for lo, hi in zip(EPI0_BINS[:-1], EPI0_BINS[1:]):
        m = (epi0 >= lo) & (epi0 < hi)
        if m.sum() < 10:
            continue
        med, err = _med_err(vals[m])
        xs.append(0.5 * (lo + hi)); ys.append(med); es.append(err); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(es), ns


def plot_bias_vs_energy(R4, RC):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    log("\n  Median mass [MeV] / median opening-angle ratio vs true pi0 energy (bins " + str(EPI0_BINS) + "):")
    for j, g in enumerate(("all", "region_true")):
        for i, (key, ekey, ylab, ref) in enumerate((
                ("masses", "epi0_m", "median reco M(gamma gamma) [MeV]", PI0_MASS_MEV),
                ("angle", "epi0_a", "median dAlpha_reco / dAlpha_mc", 1.0))):
            ax = axes[i][j]
            for R, col, ls, nm in ((R4, C_G4, "-", "Geant4"), (RC, C_CC3, "--", "CC3")):
                v = R["gates"][g][key] * (1000.0 if key == "masses" else 1.0)
                x, y, e, n = _profile(v, R["gates"][g][ekey])
                if len(x):
                    ax.errorbar(x, y, yerr=e, fmt="o", ls=ls, color=col, capsize=3, label=nm)
                    log(f"    [{g}] {key:7s} {nm:6s}: " +
                        "  ".join(f"{a:.0f}GeV:{b:.3f}(N={c})" for a, b, c in zip(x, y, n)))
            ax.axhline(ref, color="gray", ls=":")
            ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(GATE_TITLE[g], fontsize=13, fontweight="bold")
            ax.legend(framealpha=0.6)
        axes[1][j].set_xlabel("true pi0 energy E_mc1 + E_mc2 [GeV]")
    plt.tight_layout()
    fig.text(0.5, -0.005, SELECTION_FOOTER, ha="center", va="top", fontsize=10)
    fig.savefig(f"{PLOT_DIR}/bias_vs_Epi0.png", bbox_inches="tight")
    plt.close(fig)
    log("  Saved: bias_vs_Epi0.png")

# ============================================================
# PLOTS: DIAGNOSTICS PER GATE
# ============================================================
def plot_splitting(a, b, tag, gsuf):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    bins = np.arange(1.5, 9.5, 1)
    if len(a["nfrag"]):
        ax.hist(a["nfrag"], bins=bins, histtype="step", color=C_G4, linewidth=2,
                label=f"Geant4 (N={a['eff']['n_3plus']})")
    if len(b["nfrag"]):
        ax.hist(b["nfrag"], bins=bins, histtype="step", color=C_CC3, linewidth=2, linestyle="--",
                label=f"CC3 (N={b['eff']['n_3plus']})")
    ax.set_xlabel("Number of matched reco fragments per pi0")
    ax.set_ylabel("pi0 count")
    ax.set_title(f"Cluster splitting: fragment multiplicity\n{gsuf}", fontsize=11)
    if len(a["nfrag"]) or len(b["nfrag"]):
        ax.legend(framealpha=0.6)
    ax = axes[1]
    bins2 = np.linspace(0.2, 1.0, 25)
    if len(a["lead"]):
        ax.hist(a["lead"], bins=bins2, histtype="step", color=C_G4, linewidth=2, density=True, label="Geant4")
    if len(b["lead"]):
        ax.hist(b["lead"], bins=bins2, histtype="step", color=C_CC3, linewidth=2, linestyle="--",
                density=True, label="CC3")
    ax.set_xlabel("Leading fragment energy / total matched energy")
    ax.set_ylabel("Normalized density")
    ax.set_title(f"Cluster splitting: energy sharing (1.0 = one dominant fragment)\n{gsuf}", fontsize=11)
    if len(a["lead"]) or len(b["lead"]):
        ax.legend(framealpha=0.6)
    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/cluster_splitting_diagnostic{tag}.png", bbox_inches="tight")
    plt.close(fig)
    for nm, d in (("G4", a), ("CC3", b)):
        if len(d["nfrag"]):
            log(f"  Splitting [{nm}{tag}]: mean fragments {d['nfrag'].mean():.2f}, "
                f"mean leading frac {d['lead'].mean():.3f}")


def plot_response(a, b, tag, gsuf):
    if len(a["resp"]) == 0 or len(b["resp"]) == 0:
        log(f"  Energy response{tag}: SKIPPED (empty)")
        return
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.hist(a["resp"], bins=E_RESPONSE_BINS, histtype="step", color=C_G4, linewidth=2, density=True,
            label=f"Geant4 (N={len(a['resp'])}, mean={a['resp'].mean():.3f}, median={np.median(a['resp']):.3f})")
    ax.hist(b["resp"], bins=E_RESPONSE_BINS, histtype="step", color=C_CC3, linewidth=2, linestyle="--",
            density=True, label=f"CC3 (N={len(b['resp'])}, mean={b['resp'].mean():.3f}, median={np.median(b['resp']):.3f})")
    ax.axvline(1.0, color="black", lw=1.0, ls=":", alpha=0.7, label="Perfect response (1.0)")
    ax.set_xlabel("E_reco / E_mc  (per matched photon, clean n==2 pi0s)")
    ax.set_ylabel("Normalized density")
    ax.set_title(f"Photon energy response: G4 vs CC3\n{gsuf}", fontsize=11)
    ax.legend(framealpha=0.6, fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/energy_response{tag}.png", bbox_inches="tight")
    plt.close(fig)
    log(f"  Response{tag}: G4 mean/median {a['resp'].mean():.4f}/{np.median(a['resp']):.4f} | "
        f"CC3 {b['resp'].mean():.4f}/{np.median(b['resp']):.4f}")


def plot_opening_angle(a, b, tag, gsuf):
    ga, gb = a["angle"], b["angle"]
    if len(ga) == 0 or len(gb) == 0:
        log(f"  Opening angle{tag}: SKIPPED (empty)")
        return
    ca, _ = np.histogram(ga, bins=ANGLE_RATIO_BINS)
    cb, _ = np.histogram(gb, bins=ANGLE_RATIO_BINS)
    cen = 0.5 * (ANGLE_RATIO_BINS[:-1] + ANGLE_RATIO_BINS[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ca > 0, cb / ca, np.nan)
        err = ratio * np.sqrt(np.where(ca > 0, 1.0/ca, np.nan) + np.where(cb > 0, 1.0/cb, np.nan))
    lo_, hi_ = ANGLE_RATIO_BINS[0], ANGLE_RATIO_BINS[-1]
    out_a = 100 * np.mean((ga < lo_) | (ga >= hi_))
    out_b = 100 * np.mean((gb < lo_) | (gb >= hi_))
    fig = plt.figure(figsize=(9.5, 7.5))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.hist(ga, bins=ANGLE_RATIO_BINS, histtype="step", color=C_G4, linewidth=2, density=True,
            label=f"Geant4 (N={len(ga)}, median={np.median(ga):.3f}, {out_a:.1f}% outside range)")
    ax.hist(gb, bins=ANGLE_RATIO_BINS, histtype="step", color=C_CC3, linewidth=2, linestyle="--",
            density=True, label=f"CC3 (N={len(gb)}, median={np.median(gb):.3f}, {out_b:.1f}% outside range)")
    ax.axvline(1.0, color="black", lw=1.0, ls=":", alpha=0.7, label="Perfect angle (1.0)")
    ax.set_xlim(lo_, hi_)
    ax.set_ylabel("Normalized density")
    ax.set_title(f"Opening-angle bias: G4 vs CC3 (reco vs MC angle between the two photons)\n{gsuf}", fontsize=11)
    ax.legend(framealpha=0.6, fontsize=10)
    ax.tick_params(labelbottom=False)
    axr.errorbar(cen, ratio, yerr=err, fmt="o", ms=3, color="black", ecolor="gray", elinewidth=1, capsize=1.5)
    axr.axhline(1.0, color="gray", lw=1.0, ls=":")
    axr.set_xlim(lo_, hi_)
    axr.set_xlabel("dAlpha_reco / dAlpha_mc  (per clean n==2 pi0)")
    axr.set_ylabel("CC3 / G4")
    axr.set_ylim(0.0, 2.0)
    fig.savefig(f"{PLOT_DIR}/opening_angle_bias{tag}.png", bbox_inches="tight")
    plt.close(fig)
    ina, inb = ga[(ga >= lo_) & (ga < hi_)], gb[(gb >= lo_) & (gb < hi_)]
    log(f"  Opening angle{tag}: median G4 {np.median(ga):.4f} / CC3 {np.median(gb):.4f} | "
        f"in-range mean G4 {ina.mean():.4f} / CC3 {inb.mean():.4f} | outside range {out_a:.1f}% / {out_b:.1f}%")


def print_efficiency(a, b, title):
    log(f"\n{'='*66}\n  pi0 RECONSTRUCTION EFFICIENCY -- {title}\n  {'-'*62}")
    log(f"  {'Category':<30}  {'G4':>16}  {'CC3':>16}")
    rows = [("MC pi0s (denominator)", "n_mc_total", False), ("0 reco photons (lost)", "n_0reco", True),
            ("1 reco photon (merged/lost)", "n_1reco", True), ("2 reco photons (-> mass)", "n_exactly2", True),
            ("3+ or same-photon (split)", "n_3plus", True)]
    for name, k, pct in rows:
        va, vb = a["eff"][k], b["eff"][k]
        if pct:
            log(f"  {name:<30}  {va:7d} ({100*va/max(a['eff']['n_mc_total'],1):5.1f}%)  "
                f"{vb:7d} ({100*vb/max(b['eff']['n_mc_total'],1):5.1f}%)")
        else:
            log(f"  {name:<30}  {va:>16}  {vb:>16}")
    log(f"  {'Clean reco efficiency':<30}  {a['eff']['eff_pct']:15.1f}%  {b['eff']['eff_pct']:15.1f}%")
    log(f"  {'Delta (CC3-G4)':<30}  {b['eff']['eff_pct']-a['eff']['eff_pct']:+.1f} pp")
    log("  NOTE: G4 vs CC3 difference can be a composition effect (G4 sample contains photons CC3 would not fast-sim).")


def print_region_check(R):
    c = R["chk"]
    log(f"\n  Region (barrel 40-140, phi cracks removed) vs isHandledByFastSim() "
        f"[{R['label']}], pi0 daughter photons E>=10 GeV:")
    log(f"    photons E>=10 GeV: {c['n_e10']} | in geometric region: {c['n_region']} | "
        f"flagged fast-sim: {c['n_flag']} | both: {c['n_both']} | flag unavailable: {c['n_flag_unavail']}")
    log(f"    pi0s with two PFOs from the SAME daughter photon (counted as split): {c['n_same_photon']}")
    if R["label"] == "CC3" and c["n_region"] > 0:
        log(f"    CC3 flagged fraction among region photons = {100*c['n_both']/c['n_region']:.1f}%  "
            f"[single-photon sim level, not from pi0s only: 90.4%]")
        n2, n1, n0 = c["reg_pi0_2fl"], c["reg_pi0_1fl"], c["reg_pi0_0fl"]
        tot = max(n0 + n1 + n2, 1)
        log(f"    CC3 region pi0s with 2 / 1 / 0 flagged daughters: {n2} ({100*n2/tot:.1f}%) / "
            f"{n1} ({100*n1/tot:.1f}%) / {n0} ({100*n0/tot:.1f}%)   -> region_true keeps the '2' class")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 62)
    log(f"  pi0_invariant_mass_v6_3.py   run size: {RUN_SIZE} ({RUN_LABEL})")
    log(f"  Output: {PLOT_DIR}")
    log(f"  Region: barrel theta {BARREL_THETA_DEG}, phi dead window "
        f"({_PA}, {_PB}) deg from each C")
    log("  Collections: MCParticles, RecoMCTruthLink, PandoraPFOs (PDG 22)")
    log("=" * 62)
    results, todo = {}, {}
    for label, path in (("G4", G4_FILE), ("CC3", CC3_FILE)):
        r = load_cache(label)
        if r is None:
            todo[label] = path
        else:
            results[label] = r
    if todo:
        if PARALLEL and len(todo) > 1:
            with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as ex:
                futs = {lab: ex.submit(process_file, p, lab) for lab, p in todo.items()}
                for lab, f in futs.items():
                    results[lab] = f.result()
        else:
            for lab, p in todo.items():
                results[lab] = process_file(p, lab)
        for lab in todo:
            save_cache(results[lab])
    R4, RC = results["G4"], results["CC3"]

    log(f"\n  Events processed: G4 {R4['n_events']}, CC3 {RC['n_events']}")
    print_region_check(R4)
    print_region_check(RC)
    for g in GATES:
        print_efficiency(R4["gates"][g], RC["gates"][g], GATE_LONG[g])

    log("\n-- truth (zoomed) --")
    plot_truth_zoomed(R4["truth"], RC["truth"])
    log("\n-- reco mass: 4 gates x (Gaussian, Crystal Ball) --")
    plot_reco_mass(R4, RC)
    log("\n-- bias vs pi0 energy --")
    plot_bias_vs_energy(R4, RC)
    log("\n-- diagnostics per gate --")
    for g in GATES:
        gsuf = f"[{GATE_LONG[g]}]"
        a, b = R4["gates"][g], RC["gates"][g]
        plot_splitting(a, b, GATE_TAG[g], gsuf)
        plot_response(a, b, GATE_TAG[g], gsuf)
        plot_opening_angle(a, b, GATE_TAG[g], gsuf)

    with open(f"{PLOT_DIR}/pi0_summary_v6_3.txt", "w") as fh:
        fh.write(SELECTION_FOOTER + "\n\n" + "\n".join(LOG) + "\n")
    log(f"\n  All plots + pi0_summary_v6_3.txt saved to: {PLOT_DIR}\n  Done.")


if __name__ == "__main__":
    main()
