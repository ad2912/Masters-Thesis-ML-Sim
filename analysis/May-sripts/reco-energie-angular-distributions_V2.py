"""
reco_photon_distributions_v2.py
================================
Reco-level photon distributions with improved two-panel layout.
Each figure separates two distinct physics questions:
  Top panel    : G4 vs CC3 total — primary comparison
  Bottom panel : CC3 fast-sim vs CC3 G4-handled — internal CC3 composition
                 G4 drawn faintly as reference

Signal definitions:
  A. pi0 daughters : genStatus==1, E_mc>=10 GeV, parent PDG==111
  B. not-ISR       : genStatus==1, E_mc>=10 GeV, parent PDG!=11

Figures produced (8 total):
  signal_A_energy.png       — reco energy,    Signal A
  signal_A_theta.png        — reco theta,     Signal A
  signal_A_phi.png          — reco phi,       Signal A
  signal_B_energy.png       — reco energy,    Signal B
  signal_B_theta.png        — reco theta,     Signal B
  signal_B_phi.png          — reco phi,       Signal B
  signal_A_response.png     — E_reco/E_mc vs E_mc, Signal A
  signal_B_response.png     — E_reco/E_mc vs E_mc, Signal B

All distribution plots: raw counts, NOT normalized.
Energy response: mean ratio per bin (top) + RMS resolution per bin (bottom).

Run:
    source ~/source.sh
    python3 reco_photon_distributions_v2.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import edm4hep
from podio import root_io
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
#G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root"
#CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root"
G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root"
CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"

PLOT_DIR = os.path.expanduser("~/thesis-ml-sim/plots/reco_distributions_V2_twopanel_10kevents")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
PHOTON_PDG   = 22
PI0_PDG      = 111
ELECTRON_PDG = 11
E_THRESH     = 10.0
MAX_EVENTS   = 10000

# ─── Colors ───────────────────────────────────────────────────────────────────
C_G4    = "#2166ac"   # blue    — Geant4
C_CC3   = "#d6604d"   # red     — CC3 total
C_CC_FS = "#f4a582"   # orange  — CC3 fast-sim handled
C_CC_G4 = "#4dac26"   # green   — CC3 G4-handled

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 12,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 12,
    "legend.fontsize"  : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})

# ─── Binning ──────────────────────────────────────────────────────────────────
BINS_ENERGY = np.arange(0, 132, 2)
BINS_THETA  = np.linspace(0,   180, 37)
BINS_PHI    = np.linspace(-180, 180, 73)

# 5 GeV bins from 10–130 GeV: enough entries per bin for a stable mean
# at 1000-event sample size (~50–100 photons/bin at peak)
BINS_RESP   = np.arange(10, 131, 5)

# ─── Detector boundary markers ────────────────────────────────────────────────
# Theta: ILD barrel-endcap transitions + forward region edges
THETA_VLINES = [10.0, 35.0, 40.0, 140.0, 145.0, 170.0]

# Phi: ILD ECal barrel octagon — 8 module boundaries, equally spaced every 45°
# The octagon faces are centred away from φ=0; boundaries start at
# φ = -180° + 22.5° = -157.5°, then every 45°.
# This gives 8 lines: -157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5
PHI_VLINES = [-157.5 + 45.0 * i for i in range(8)]


# ─── Loader ───────────────────────────────────────────────────────────────────
def load_file(label, path, is_cc3=False):
    """
    Single pass through reco file.
    For each PandoraPFO with PDG==22, follows RecoMCTruthLink and applies
    base cuts: genStatus==1, MC PDG==22, E_mc >= 10 GeV.
    Reco kinematics extracted from edm4hep.utils.p4(pfo).
    Signal cuts (parentage, isHandledByFastSim) applied downstream via masks.
    """
    print(f"  Loading {label} ...")

    if not os.path.exists(path):
        print(f"  ERROR: not found: {path}")
        return None

    reader = root_io.Reader(path)

    reco_energy_list = []
    reco_theta_list  = []
    reco_phi_list    = []
    mc_energy_list   = []
    is_pi0_list      = []
    is_isr_list      = []
    is_fs_list       = []

    n_events = 0
    for event in reader.get("events"):
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        try:
            reco_to_mc = {}
            for link in event.get("RecoMCTruthLink"):
                reco = link.getFrom()
                mc   = link.getTo()
                reco_to_mc[reco.getObjectID().index] = mc
        except Exception:
            continue

        try:
            pfos = list(event.get("PandoraPFOs"))
        except Exception:
            continue

        for pfo in pfos:
            if pfo.getPDG() != PHOTON_PDG:
                continue

            mc = reco_to_mc.get(pfo.getObjectID().index, None)
            if mc is None:
                continue
            if mc.getGeneratorStatus() != 1:
                continue
            if mc.getPDG() != PHOTON_PDG:
                continue
            if mc.getEnergy() < E_THRESH:
                continue

            p4    = edm4hep.utils.p4(pfo)
            theta = np.degrees(float(p4.theta()))
            phi   = np.degrees(float(p4.phi()))

            parent_pdgs = [abs(par.getPDG()) for par in mc.getParents()]

            reco_energy_list.append(pfo.getEnergy())
            reco_theta_list.append(theta)
            reco_phi_list.append(phi)
            mc_energy_list.append(mc.getEnergy())
            is_pi0_list.append(PI0_PDG      in parent_pdgs)
            is_isr_list.append(ELECTRON_PDG in parent_pdgs)
            is_fs_list.append(bool(mc.isHandledByFastSim()) if is_cc3 else False)

    d = {
        "reco_energy": np.array(reco_energy_list),
        "reco_theta" : np.array(reco_theta_list),
        "reco_phi"   : np.array(reco_phi_list),
        "mc_energy"  : np.array(mc_energy_list),
        "is_pi0"     : np.array(is_pi0_list, dtype=bool),
        "is_isr"     : np.array(is_isr_list, dtype=bool),
        "is_fs"      : np.array(is_fs_list,  dtype=bool),
        "n_events"   : n_events,
    }

    n = len(d["reco_energy"])
    print(f"    {n_events} events | {n} base photons (genStatus==1, E_mc>={E_THRESH} GeV)")
    print(f"    pi0 daughters : {d['is_pi0'].sum()}"
          f"  |  not-ISR : {(~d['is_isr']).sum()}"
          + (f"  |  fast-sim : {d['is_fs'].sum()}" if is_cc3 else ""))
    return d


# ─── Detector boundary helpers ────────────────────────────────────────────────
def add_detector_lines(ax, observable):
    """Draw physically motivated detector boundary markers."""
    if observable == "theta":
        for i, xv in enumerate(THETA_VLINES):
            lbl = "Detector boundaries" if i == 0 else None
            ax.axvline(xv, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, label=lbl)
    elif observable == "phi":
        for i, xv in enumerate(PHI_VLINES):
            lbl = "ECal module boundaries (45° offset)" if i == 0 else None
            ax.axvline(xv, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, label=lbl)
    elif observable == "energy":
        ax.axvline(E_THRESH, color="black", linewidth=1.0, linestyle=":",
                   alpha=0.7, label=f"CC3 threshold ({E_THRESH:.0f} GeV)")


# ─── Two-panel distribution plot ──────────────────────────────────────────────
def make_two_panel_plot(observable, bins, signal_label, filename,
                        g4_vals, cc3_total_vals,
                        cc3_fs_vals, cc3_g4h_vals,
                        n_g4, n_cc3, xlabel):
    """
    Top panel    : G4 vs CC3 total
                   Physics question: do the two simulations agree at reco level?

    Bottom panel : CC3 fast-sim vs CC3 G4-handled, with G4 as faint reference
                   Physics question: what is the internal composition of CC3,
                   and where does each component sit relative to full G4?
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [1, 1]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.08)

    # ── Top: G4 vs CC3 total ──────────────────────────────────────────────────
    ax_top.hist(g4_vals, bins=bins, histtype="step",
                color=C_G4, linewidth=2,
                label=f"G4 (N={len(g4_vals)}, {n_g4} ev)")
    ax_top.hist(cc3_total_vals, bins=bins, histtype="step",
                color=C_CC3, linewidth=2, linestyle="--",
                label=f"CC3 total (N={len(cc3_total_vals)}, {n_cc3} ev)")

    add_detector_lines(ax_top, observable)
    if observable == "energy":
        ax_top.set_yscale("log")

    ax_top.set_ylabel("Photons / bin  [raw counts]")
    ax_top.set_title(f"Reco {observable.upper()} — {signal_label} | G4 vs CC3")
    ax_top.legend(framealpha=0.5, fontsize=10, loc="upper right")
    ax_top.text(0.01, 1.02,
                "ILD reco — 1000 events  |  raw counts  |  reco kinematics, MC truth signal cuts",
                transform=ax_top.transAxes, fontsize=8, color="gray", va="bottom")

    # ── Bottom: CC3 composition + G4 faint reference ──────────────────────────
    ax_bot.hist(g4_vals, bins=bins, histtype="stepfilled",
                color=C_G4, alpha=0.12,
                label=f"G4 reference (N={len(g4_vals)})")
    ax_bot.hist(g4_vals, bins=bins, histtype="step",
                color=C_G4, linewidth=1.2, linestyle="--", alpha=0.4)

    ax_bot.hist(cc3_fs_vals, bins=bins, histtype="stepfilled",
                color=C_CC_FS, alpha=0.6,
                label=f"CC3 fast-sim (N={len(cc3_fs_vals)})")
    ax_bot.hist(cc3_fs_vals, bins=bins, histtype="step",
                color=C_CC_FS, linewidth=1.8)

    ax_bot.hist(cc3_g4h_vals, bins=bins, histtype="step",
                color=C_CC_G4, linewidth=2, linestyle="-.",
                label=f"CC3 G4-handled (N={len(cc3_g4h_vals)})")

    add_detector_lines(ax_bot, observable)
    if observable == "energy":
        ax_bot.set_yscale("log")

    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel("Photons / bin  [raw counts]")
    ax_bot.set_title("CC3 internal composition  (fast-sim + G4-handled = CC3 total)")
    ax_bot.legend(framealpha=0.5, fontsize=10, loc="upper right")

    plt.tight_layout()
    out = f"{PLOT_DIR}/{filename}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {filename}")


# ─── Energy response: profile computation ────────────────────────────────────
def compute_response_profile(mc_e, reco_e, bins):
    """
    For each bin in mc_e, compute mean(reco_e / mc_e) and RMS (= resolution).
    Returns (bin_centers, means, rmss, n_per_bin).
    Bins with fewer than 3 entries are set to NaN and skipped in plotting.

    Physics: mean < 1.0 indicates energy underestimation (scale bias).
             RMS measures the per-bin spread of the response (energy resolution).
    """
    ratio   = reco_e / mc_e
    centers = 0.5 * (bins[:-1] + bins[1:])
    means, rmss, ns = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (mc_e >= lo) & (mc_e < hi)
        r = ratio[mask]
        if len(r) >= 3:
            means.append(np.mean(r))
            rmss.append(np.std(r))
            ns.append(len(r))
        else:
            means.append(np.nan)
            rmss.append(np.nan)
            ns.append(0)

    return centers, np.array(means), np.array(rmss), np.array(ns)


# ─── Energy response: two-panel plot ─────────────────────────────────────────
def make_response_plot(signal_label, filename,
                       g4_mc,      g4_reco,
                       cc3_mc,     cc3_reco,
                       cc3_fs_mc,  cc3_fs_reco,
                       cc3_g4h_mc, cc3_g4h_reco):
    """
    Two-panel energy response plot.

    Top panel    : mean(E_reco / E_mc) vs E_mc
                   Physics question: is there an energy scale bias, and does it
                   differ between G4 and CC3?

    Bottom panel : RMS of (E_reco / E_mc) per bin — energy resolution
                   Physics question: does CC3 produce better or worse energy
                   resolution than G4, as a function of true photon energy?

    Separation into two panels avoids the visual confusion of wide error bars
    overlapping four datasets on the same axes.
    """
    fig, (ax_mean, ax_rms) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 2]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.08)

    datasets = [
        (g4_mc,      g4_reco,      C_G4,    "G4",             "o",  2.0, "-"),
        (cc3_mc,     cc3_reco,     C_CC3,   "CC3 total",      "s",  1.5, "--"),
        (cc3_fs_mc,  cc3_fs_reco,  C_CC_FS, "CC3 fast-sim",   "^",  1.5, "-"),
        (cc3_g4h_mc, cc3_g4h_reco, C_CC_G4, "CC3 G4-handled", "v",  1.5, "-."),
    ]

    for mc_e, reco_e, color, label, marker, lw, ls in datasets:
        if len(mc_e) < 5:
            continue
        centers, means, rmss, ns = compute_response_profile(mc_e, reco_e, BINS_RESP)
        valid = ~np.isnan(means)

        ax_mean.plot(centers[valid], means[valid],
                     marker=marker, color=color, linewidth=lw, linestyle=ls,
                     markersize=5, label=f"{label} (N={len(mc_e)})")

        ax_rms.plot(centers[valid], rmss[valid],
                    marker=marker, color=color, linewidth=lw, linestyle=ls,
                    markersize=5)

    # Perfect response reference — only on mean panel
    ax_mean.axhline(1.0, color="black", linewidth=1.0, linestyle=":",
                    alpha=0.7, label="Perfect response")

    ax_mean.set_ylabel("Mean  E_reco / E_mc")
    ax_mean.set_title(f"Energy Response — {signal_label}")
    ax_mean.set_ylim(0.4, 1.4)
    ax_mean.legend(framealpha=0.5, fontsize=9, loc="lower right")
    ax_mean.text(0.01, 1.02,
                 "ILD reco — 1000 events  |  reco kinematics, MC truth signal cuts",
                 transform=ax_mean.transAxes, fontsize=8, color="gray", va="bottom")

    ax_rms.set_xlabel("MC Truth Photon Energy [GeV]")
    ax_rms.set_ylabel("RMS  E_reco / E_mc  (resolution)")
    ax_rms.set_title("Energy Resolution per bin")
    ax_rms.set_xlim(BINS_RESP[0], BINS_RESP[-1])
    ax_rms.set_ylim(bottom=0)

    plt.tight_layout()
    out = f"{PLOT_DIR}/{filename}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {filename}")


# ─── Main ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Reco photon distributions v2 — two-panel layout")
print("=" * 60 + "\n")

g4  = load_file("G4  (full sim)", G4_FILE,  is_cc3=False)
cc3 = load_file("CC3 (fast sim)", CC3_FILE, is_cc3=True)

if not g4 or not cc3:
    raise SystemExit("ERROR: could not load one or both files.")

print()

# ─── Signal masks ─────────────────────────────────────────────────────────────
g4_A      = g4["is_pi0"]
cc3_A     = cc3["is_pi0"]
cc3_A_fs  = cc3_A &  cc3["is_fs"]
cc3_A_g4h = cc3_A & ~cc3["is_fs"]

g4_B      = ~g4["is_isr"]
cc3_B     = ~cc3["is_isr"]
cc3_B_fs  = cc3_B &  cc3["is_fs"]
cc3_B_g4h = cc3_B & ~cc3["is_fs"]

# ─── Partition consistency check ──────────────────────────────────────────────
# fast-sim and G4-handled must be mutually exclusive and exhaustive within
# each signal selection. If this fires, isHandledByFastSim() or the masking
# logic is broken.
for name, total_mask, fs_mask, g4h_mask in [
    ("Signal A", cc3_A, cc3_A_fs, cc3_A_g4h),
    ("Signal B", cc3_B, cc3_B_fs, cc3_B_g4h),
]:
    n_total = total_mask.sum()
    n_fs    = fs_mask.sum()
    n_g4h   = g4h_mask.sum()
    assert n_fs + n_g4h == n_total, (
        f"CC3 partition broken for {name}: "
        f"fast-sim({n_fs}) + G4-handled({n_g4h}) = {n_fs+n_g4h} != total({n_total})"
    )
    print(f"  ✓  {name} partition check: {n_fs} FS + {n_g4h} G4h = {n_total} total")

print()

n_g4  = g4["n_events"]
n_cc3 = cc3["n_events"]

SIG_A = "genStatus==1, E_mc≥10 GeV, π⁰ daughter"
SIG_B = "genStatus==1, E_mc≥10 GeV, not ISR"

# ─── Distribution plots ───────────────────────────────────────────────────────
for sig_label, sig_tag, g4_mask, cc3_mask, cc3_fs_mask, cc3_g4h_mask in [
    (SIG_A, "signal_A", g4_A, cc3_A, cc3_A_fs, cc3_A_g4h),
    (SIG_B, "signal_B", g4_B, cc3_B, cc3_B_fs, cc3_B_g4h),
]:
    print(f"{sig_tag} ({sig_label}):")

    for observable, bins, xlabel in [
        ("energy", BINS_ENERGY, "Reco Photon Energy [GeV]"),
        ("theta",  BINS_THETA,  "Reco Photon θ [deg]"),
        ("phi",    BINS_PHI,    "Reco Photon φ [deg]"),
    ]:
        key = f"reco_{observable}"
        make_two_panel_plot(
            observable     = observable,
            bins           = bins,
            signal_label   = sig_label,
            filename       = f"{sig_tag}_{observable}.png",
            g4_vals        = g4[key][g4_mask],
            cc3_total_vals = cc3[key][cc3_mask],
            cc3_fs_vals    = cc3[key][cc3_fs_mask],
            cc3_g4h_vals   = cc3[key][cc3_g4h_mask],
            n_g4=n_g4, n_cc3=n_cc3,
            xlabel=xlabel,
        )
    print()

# ─── Energy response plots ────────────────────────────────────────────────────
print("Energy response (two-panel: mean + resolution):")

for sig_label, sig_tag, g4_mask, cc3_mask, cc3_fs_mask, cc3_g4h_mask in [
    (SIG_A, "signal_A", g4_A, cc3_A, cc3_A_fs, cc3_A_g4h),
    (SIG_B, "signal_B", g4_B, cc3_B, cc3_B_fs, cc3_B_g4h),
]:
    make_response_plot(
        signal_label = sig_label,
        filename     = f"{sig_tag}_response.png",
        g4_mc        = g4["mc_energy"][g4_mask],
        g4_reco      = g4["reco_energy"][g4_mask],
        cc3_mc       = cc3["mc_energy"][cc3_mask],
        cc3_reco     = cc3["reco_energy"][cc3_mask],
        cc3_fs_mc    = cc3["mc_energy"][cc3_fs_mask],
        cc3_fs_reco  = cc3["reco_energy"][cc3_fs_mask],
        cc3_g4h_mc   = cc3["mc_energy"][cc3_g4h_mask],
        cc3_g4h_reco = cc3["reco_energy"][cc3_g4h_mask],
    )

print(f"\n  All 8 plots saved to: {PLOT_DIR}\n")
