#!/usr/bin/env python3
"""
event_display.py — ILD ECal + Tracker hit display for photon gun events
Physics question: Where are the ECal hits spatially, and do we see
tracker activity that would indicate photon conversion before the ECal?

Three panels per event:
  Left  : Transverse view (x-y), shows azimuthal shower shape
  Middle: Longitudinal view (z-r), shows shower depth and entry point
  Right : Lego view (eta-phi), standard HEP shower footprint

Tracker hits overlaid in colour-coded layers.
ECal hits sized by sqrt(E_dep) and coloured by E_dep.

Usage:
  python event_display.py                    # display event 0 for all files
  python event_display.py --event 42         # specific event index
  python event_display.py --find-empty       # auto-find first empty-ECal event
  python event_display.py --find-split       # auto-find a candidate split event

Author: alimuham @ DESY
"""

import os, math, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────
#  Paths — same structure as analysis_v3.py
# ─────────────────────────────────────────────
DATA_DIR = "/data/dust/user/alimuham/thesis/cc3_low_E_data"
OUT_DIR  = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/cc3_low_E_plots/event_displays"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "G4"          : os.path.join(DATA_DIR, "photon_gun_1to10GeV_g4_z20mm.edm4hep.root"),
    "CC3 original": os.path.join(DATA_DIR, "photon_gun_1to10GeV_cc3_original_z20mm.edm4hep.root"),
    "CC3 630K"    : os.path.join(DATA_DIR, "photon_gun_1to10GeV_cc3_630K_z20mm.edm4hep.root"),
}

# ─────────────────────────────────────────────
#  Collection definitions
# ─────────────────────────────────────────────
# ECal: barrel + endcap Si layers (Sc layers tend to be empty for photon gun,
# but included via try/except so no harm adding them)
ECAL_COLLS = [
    "ECalBarrelSiHitsEven", "ECalBarrelSiHitsOdd",
    "ECalBarrelScHitsEven", "ECalBarrelScHitsOdd",
    "ECalEndcapSiHitsEven", "ECalEndcapSiHitsOdd",
    "ECalEndcapScHitsEven", "ECalEndcapScHitsOdd",
    "EcalEndcapRingCollection",
]

# Tracker: name → (display colour, marker size, z-order)
# Physics note: if a photon converts (γ → e+e-) in the tracker material,
# we expect hits here with NO corresponding ECal barrel shower.
# Colour choice: cool colours for tracker (background), hot colourmap for ECal (foreground).
TRACKER_COLLS = {
    "VXD" : ("VXDCollection",       "#00cfff", 3, 6),   # cyan  — innermost silicon vertex
    "SIT" : ("SITCollection",       "#00e676", 3, 6),   # green — silicon inner tracker
    "TPC" : ("TPCCollection",       "#76ff03", 2, 5),   # lime  — TPC (many hits, small markers)
    "SET" : ("SETCollection",       "#ffea00", 4, 7),   # yellow — silicon envelope, just before ECal
}


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def _gun_energy(frame):
    """Return gun photon |p| in GeV. genStatus==1 and PDG==22 selects the primary."""
    for p in frame.get("MCParticles"):
        if abs(p.getPDG()) == 22 and p.getGeneratorStatus() == 1:
            m = p.getMomentum()
            return math.sqrt(m.x**2 + m.y**2 + m.z**2)
    return float("nan")


def _collect_ecal_hits(frame):
    """
    Return arrays (x, y, z, e) for all ECal hits in this event.
    Uses try/except per collection — missing collections are silently skipped.
    This handles both barrel-only and barrel+endcap files transparently.
    """
    xs, ys, zs, es = [], [], [], []
    for cname in ECAL_COLLS:
        try:
            for h in frame.get(cname):
                pos = h.getPosition()
                xs.append(pos.x); ys.append(pos.y)
                zs.append(pos.z); es.append(h.getEnergy())
        except Exception:
            pass
    return np.array(xs), np.array(ys), np.array(zs), np.array(es)


def _collect_tracker_hits(frame):
    """
    Return dict: tracker_label → (x, y, z arrays, colour, markersize, zorder).
    getEDep() exists on SimTrackerHit but is tiny (MIP deposit) — not used for sizing.
    """
    result = {}
    for tlabel, (cname, color, ms, zo) in TRACKER_COLLS.items():
        xs, ys, zs = [], [], []
        try:
            for h in frame.get(cname):
                pos = h.getPosition()
                xs.append(pos.x); ys.append(pos.y); zs.append(pos.z)
        except Exception:
            pass
        result[tlabel] = (np.array(xs), np.array(ys), np.array(zs), color, ms, zo)
    return result


# ─────────────────────────────────────────────
#  Event finders
# ─────────────────────────────────────────────
def find_empty_ecal_event(path, n_scan=200):
    """
    Scan up to n_scan events and return the index of the first event
    with zero ECal hits. These are the "mystery" events your supervisor
    flagged — showing tracker activity (or none) for these is the key diagnostic.
    """
    import podio.root_io as rio
    r = rio.Reader(path)
    for i, frame in enumerate(r.get("events")):
        if i >= n_scan:
            break
        xs, _, _, _ = _collect_ecal_hits(frame)
        if len(xs) == 0:
            print(f"[FINDER] First empty ECal event: idx={i}")
            return i
    print("[FINDER] No empty ECal event found in first {n_scan} events — returning 0")
    return 0


def find_split_candidate_event(path, n_scan=300):
    """
    Scan events and return index of one that likely shows cluster splitting.
    Heuristic: ECal hits present, but the hit phi distribution has large std
    (two spatially separated clusters → bimodal phi → large std).
    Expected: std > 0.15 rad suggests two clusters ~17 deg apart or more.
    This is a heuristic only — verify by eye after plotting.
    """
    import podio.root_io as rio
    r = rio.Reader(path)
    best_idx, best_std = 0, 0.0
    for i, frame in enumerate(r.get("events")):
        if i >= n_scan:
            break
        xs, ys, zs, es = _collect_ecal_hits(frame)
        if len(xs) < 10:
            continue
        phis = np.arctan2(ys, xs)
        std = float(np.std(phis))
        if std > best_std:
            best_std = std
            best_idx = i
    print(f"[FINDER] Best split candidate: idx={best_idx}, phi_std={best_std:.3f} rad")
    return best_idx


# ─────────────────────────────────────────────
#  Core display function
# ─────────────────────────────────────────────
def make_event_display(label, path, event_idx=0, tag=""):
    """
    Three-panel event display for one event:
      Panel 1 — Transverse (x-y): azimuthal shower shape, sees phi-cracks in ECAL
      Panel 2 — Longitudinal (z-r): shower depth, entry point, endcap vs barrel
      Panel 3 — Lego (eta-phi): standard HEP footprint, natural for PandoraPFA comparison

    ECal hits: coloured by E_dep [GeV], sized by sqrt(E_dep) normalised to [4, 80] pt.
    Tracker hits: fixed small markers, colour per subdetector.
    Gun direction (1,0,0) drawn as dashed line in panels 1 and 2.
    """
    import podio.root_io as rio

    if not os.path.exists(path):
        print(f"[SKIP] {label}: file not found")
        return

    events = list(rio.Reader(path).get("events"))
    if event_idx >= len(events):
        print(f"[SKIP] Event {event_idx} out of range (file has {len(events)} events)")
        return

    frame   = events[event_idx]
    E_gun   = _gun_energy(frame)
    ecal_x, ecal_y, ecal_z, ecal_e = _collect_ecal_hits(frame)
    trk     = _collect_tracker_hits(frame)

    n_ecal    = len(ecal_x)
    n_trk     = sum(len(v[0]) for v in trk.values())
    ecal_etot = float(ecal_e.sum()) if n_ecal > 0 else 0.0

    print(f"[DISPLAY] {label} evt={event_idx}  E_gun={E_gun:.2f} GeV  "
          f"ECal hits={n_ecal}  tracker hits={n_trk}  E_dep_total={ecal_etot:.4f} GeV")

    # Derived quantities for ECal
    if n_ecal > 0:
        ecal_r   = np.sqrt(ecal_x**2 + ecal_y**2)
        ecal_phi = np.arctan2(ecal_y, ecal_x)
        # eta = -ln(tan(theta/2)), theta = angle from z-axis
        ecal_theta = np.arctan2(ecal_r, ecal_z)
        # Guard against theta=0 or pi
        ecal_theta = np.clip(ecal_theta, 1e-6, np.pi - 1e-6)
        ecal_eta  = -np.log(np.tan(ecal_theta / 2.0))
        # Marker sizes: sqrt(E) scaled to [4, 80] pt
        sizes = np.sqrt(ecal_e)
        if sizes.max() > 0:
            sizes = 4 + 76 * (sizes / sizes.max())
        else:
            sizes = np.full_like(ecal_e, 10.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cmap = cm.get_cmap("hot")

    # ── Helper: draw tracker hits on an axis ──────────────────────────────
    def _draw_tracker(ax, get_x, get_y):
        """get_x, get_y: callables(txs, tys, tzs) → x-coord, y-coord arrays"""
        for tlabel, (txs, tys, tzs, color, ms, zo) in trk.items():
            if len(txs) == 0:
                continue
            ax.scatter(get_x(txs, tys, tzs), get_y(txs, tys, tzs),
                       s=ms**2, color=color, alpha=0.6, zorder=zo,
                       label=f"{tlabel} ({len(txs)})")

    # ── Panel 1: Transverse (x-y) ─────────────────────────────────────────
    ax = axes[0]
    _draw_tracker(ax,
                  get_x=lambda x,y,z: x,
                  get_y=lambda x,y,z: y)
    if n_ecal > 0:
        norm_e = (ecal_e - ecal_e.min()) / (ecal_e.max() - ecal_e.min() + 1e-12)
        sc = ax.scatter(ecal_x, ecal_y, s=sizes,
                        c=ecal_e, cmap="hot", vmin=0, vmax=ecal_e.max(),
                        zorder=10, alpha=0.9)
        plt.colorbar(sc, ax=ax, label="E_dep [GeV]", fraction=0.046, pad=0.04)

    # Gun direction (1,0,0) from origin — in x-y plane this is the +x axis
    ax_lim = max(2500, (np.sqrt(ecal_x**2 + ecal_y**2).max() * 1.2) if n_ecal > 0 else 2500)
    ax.annotate("", xy=(ax_lim, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5, linestyle="dashed"))
    ax.set_xlim(-ax_lim * 0.1, ax_lim)
    ax.set_ylim(-ax_lim * 0.5, ax_lim * 0.5)
    ax.set_facecolor("#0d0d0d")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"Transverse (x-y)\nN_ECal={n_ecal}, E_dep={ecal_etot:.3f} GeV")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.4)

    # ── Panel 2: Longitudinal (z-r) ───────────────────────────────────────
    ax = axes[1]
    _draw_tracker(ax,
                  get_x=lambda x,y,z: z,
                  get_y=lambda x,y,z: np.sqrt(x**2 + y**2))
    if n_ecal > 0:
        sc2 = ax.scatter(ecal_z, ecal_r, s=sizes,
                         c=ecal_e, cmap="hot", vmin=0, vmax=ecal_e.max(),
                         zorder=10, alpha=0.9)
        plt.colorbar(sc2, ax=ax, label="E_dep [GeV]", fraction=0.046, pad=0.04)

    ax.set_facecolor("#0d0d0d")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("r [mm]")
    ax.set_title("Longitudinal (z-r)\nbarrel: r≈1843 mm | endcap: |z|≈2350 mm")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.4)

    # ── Panel 3: Lego (eta-phi) ───────────────────────────────────────────
    ax = axes[2]
    if n_ecal > 0:
        sc3 = ax.scatter(ecal_eta, ecal_phi, s=sizes,
                         c=ecal_e, cmap="hot", vmin=0, vmax=ecal_e.max(),
                         zorder=10, alpha=0.9)
        plt.colorbar(sc3, ax=ax, label="E_dep [GeV]", fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No ECal hits", transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=14)

    ax.set_facecolor("#0d0d0d")
    ax.set_xlabel("η (pseudorapidity)"); ax.set_ylabel("φ [rad]")
    ax.set_title("Lego view (η-φ)\nstandard HEP shower footprint")

    # ── Shared tracker legend for lego panel ─────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f"{tl} ({len(v[0])} hits)",
               markerfacecolor=v[3], markersize=6)
        for tl, v in trk.items() if len(v[0]) > 0
    ]
    if legend_elements:
        axes[2].legend(handles=legend_elements, fontsize=7, loc="upper right", framealpha=0.4)

    # ── Figure title ──────────────────────────────────────────────────────
    status = "EMPTY ECAL" if n_ecal == 0 else f"E_dep={ecal_etot:.3f} GeV"
    fig.patch.set_facecolor("#1a1a1a")
    for a in axes:
        a.tick_params(colors="white")
        a.xaxis.label.set_color("white")
        a.yaxis.label.set_color("white")
        a.title.set_color("white")
        for spine in a.spines.values():
            spine.set_edgecolor("#444")

    fig.suptitle(
        f"{label}  |  Event {event_idx}  |  E_gun = {E_gun:.2f} GeV  |  {status}\n"
        f"Gun: pos=(0,0,20) mm, dir=(1,0,0)  |  ILD_l5_v02  |  Marker size ∝ √E_dep",
        color="white", fontsize=10
    )
    fig.tight_layout()

    suffix = f"_{tag}" if tag else ""
    fname  = f"evtdisp_{label.replace(' ','_')}_evt{event_idx:04d}{suffix}.png"
    fpath  = os.path.join(OUT_DIR, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[SAVED] {fpath}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ILD ECal event display")
    parser.add_argument("--event",       type=int, default=0,
                        help="Event index to display (default: 0)")
    parser.add_argument("--find-empty",  action="store_true",
                        help="Auto-find first empty-ECal event in G4 file")
    parser.add_argument("--find-split",  action="store_true",
                        help="Auto-find best split-candidate event in CC3 original file")
    parser.add_argument("--all-files",   action="store_true",
                        help="Run on all three files (G4, CC3 original, CC3 630K)")
    args = parser.parse_args()

    event_idx = args.event

    if args.find_empty:
        event_idx = find_empty_ecal_event(FILES["G4"])
        print(f"[INFO] Using empty-ECal event: {event_idx}")
        for label, path in FILES.items():
            make_event_display(label, path, event_idx=event_idx, tag="empty_ecal")

    elif args.find_split:
        event_idx = find_split_candidate_event(FILES["CC3 original"])
        print(f"[INFO] Using split candidate event: {event_idx}")
        for label, path in FILES.items():
            make_event_display(label, path, event_idx=event_idx, tag="split_candidate")

    elif args.all_files:
        for label, path in FILES.items():
            make_event_display(label, path, event_idx=event_idx)

    else:
        # Default: show the requested event for G4 only
        make_event_display("G4", FILES["G4"], event_idx=event_idx)
