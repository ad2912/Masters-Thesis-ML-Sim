"""
split_cluster_diagnostic.py
============================
Investigates why reco photons with E_mc >= 10 GeV are reconstructed
below 10 GeV at reco level.

Physics question:
  Are these "Group B" photons explained by PandoraPFA splitting one
  shower into multiple PFOs that share the same MC parent photon?

Signal: Signal B (genStatus==1, E_mc>=10 GeV, parentPDG != 11)

Run:
    source ~/source.sh
    python3 split_cluster_diagnostic.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io
import math
import os
from collections import defaultdict, Counter

# ── CONFIG ───────────────────────────────────────────────────────────────────
G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root"
CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"
#G4_FILE  = "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root"
#CC3_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root"

PLOT_DIR   = os.path.expanduser("~/thesis-ml-sim/plots/split_diagnostic_10k")
os.makedirs(PLOT_DIR, exist_ok=True)

PHOTON_PDG    = 22
ELECTRON_PDG  = 11
E_MC_MIN      = 10.0   # GeV
MAX_EVENTS    = 10000
SPLIT_DA_MAX  = 0.1    # rad — max angular sep to call a split candidate
SPLIT_ESUM_TOL = 0.35  # fractional: |sum(E_reco) - E_mc| / E_mc < this

# ── HELPERS ──────────────────────────────────────────────────────────────────

def delta_alpha(mom1, mom2):
    """3D opening angle in radians between two momentum 3-vectors."""
    px1, py1, pz1 = mom1.x, mom1.y, mom1.z
    px2, py2, pz2 = mom2.x, mom2.y, mom2.z
    mag1 = math.sqrt(px1**2 + py1**2 + pz1**2)
    mag2 = math.sqrt(px2**2 + py2**2 + pz2**2)
    if mag1 == 0 or mag2 == 0:
        return float('nan')
    cos_a = (px1*px2 + py1*py2 + pz1*pz2) / (mag1 * mag2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.acos(cos_a)

def get_parent_pdg(mc_part):
    parents = mc_part.getParents()
    if len(parents) == 0:
        return 0
    return abs(parents[0].getPDG())

# ── MAIN DIAGNOSTIC ──────────────────────────────────────────────────────────

def run_diagnostic(filepath, label, is_cc3=False):
    print(f"\n{'='*60}")
    print(f"  {label}  —  split cluster diagnostic")
    print(f"{'='*60}")

    reader = root_io.Reader(filepath)
    records = []
    n_groupA = 0
    n_groupB = 0

    for i_ev, event in enumerate(reader.get("events")):
        if i_ev >= MAX_EVENTS:
            break

        # ── Build reco→MC map (highest weight per reco) ──────────────────
        reco_to_mc = {}
        for link in event.get("RecoMCTruthLink"):
            reco    = link.getFrom()
            mc      = link.getTo()
            weight  = link.getWeight()
            rid     = reco.getObjectID().index
            if rid not in reco_to_mc or weight > reco_to_mc[rid][1]:
                reco_to_mc[rid] = (mc, weight)

        # ── Build inverted MC→reco map (all reco per MC, any weight) ─────
        mc_to_recos = defaultdict(list)
        for link in event.get("RecoMCTruthLink"):
            reco   = link.getFrom()
            mc     = link.getTo()
            mc_id  = mc.getObjectID().index
            mc_to_recos[mc_id].append(reco)

        # ── Loop PFOs ────────────────────────────────────────────────────
        for pfo in event.get("PandoraPFOs"):
            if pfo.getPDG() != PHOTON_PDG:
                continue

            rid = pfo.getObjectID().index
            if rid not in reco_to_mc:
                continue
            mc, _ = reco_to_mc[rid]

            # Signal B truth cuts
            if mc.getGeneratorStatus() != 1:
                continue
            if mc.getEnergy() < E_MC_MIN:
                continue
            if get_parent_pdg(mc) == ELECTRON_PDG:
                continue  # ISR rejection

            e_reco = pfo.getEnergy()
            e_mc   = mc.getEnergy()
            is_fs  = bool(mc.isHandledByFastSim()) if is_cc3 else False

            if e_reco >= E_MC_MIN:
                n_groupA += 1
                continue

            # ── Group B photon ───────────────────────────────────────────
            n_groupB += 1
            mc_id = mc.getObjectID().index

            # All reco photons sharing this MC photon, excluding self
            all_partners = [r for r in mc_to_recos.get(mc_id, [])
                            if r.getObjectID().index != rid]

            # Compute partner info
            partner_info = []
            pfo_mom = pfo.getMomentum()
            for partner in all_partners:
                p_mom  = partner.getMomentum()
                da     = delta_alpha(pfo_mom, p_mom)
                e_part = partner.getEnergy()
                partner_info.append({"e": e_part, "da": da})

            # Energy sum: self + all partners
            e_sum      = e_reco + sum(p["e"] for p in partner_info)
            e_sum_frac = abs(e_sum - e_mc) / e_mc if e_mc > 0 else float('nan')

            # Best partner = highest energy among partners
            best = max(partner_info, key=lambda x: x["e"]) if partner_info else None

            # Classify
            if best is not None:
                da_ok   = best["da"] < SPLIT_DA_MAX
                esum_ok = e_sum_frac < SPLIT_ESUM_TOL
                if da_ok and esum_ok:
                    cls = "split_confirmed"
                elif da_ok:
                    cls = "split_close_badsum"
                elif esum_ok:
                    cls = "split_faraway_goodsum"
                else:
                    cls = "split_partial"
            else:
                cls = "unexplained_no_partner"

            records.append({
                "e_reco":       e_reco,
                "e_mc":         e_mc,
                "is_fs":        is_fs,
                "n_partners":   len(partner_info),
                "best_da":      best["da"]  if best else float('nan'),
                "best_e_part":  best["e"]   if best else float('nan'),
                "e_sum_frac":   e_sum_frac,
                "cls":          cls,
            })

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n  Group A (E_reco >= {E_MC_MIN} GeV): {n_groupA}")
    print(f"  Group B (E_reco <  {E_MC_MIN} GeV): {n_groupB}")
    if n_groupB == 0:
        print("  No Group B photons found — check cuts.")
        return records

    print(f"\n  Group B classification:")
    counts = Counter(r["cls"] for r in records)
    for cls in ["split_confirmed","split_close_badsum",
                "split_faraway_goodsum","split_partial","unexplained_no_partner"]:
        n = counts.get(cls, 0)
        print(f"    {cls:<30}: {n:4d}  ({100*n/n_groupB:.1f}%)")

    if is_cc3:
        fs_recs  = [r for r in records if r["is_fs"]]
        g4h_recs = [r for r in records if not r["is_fs"]]
        for subset, name in [(fs_recs, "CC3 fast-sim"), (g4h_recs, "CC3 G4-handled")]:
            if len(subset) == 0:
                continue
            n_split = sum(1 for r in subset if r["cls"] == "split_confirmed")
            print(f"\n  {name}: {len(subset)} Group B photons | "
                  f"split_confirmed: {n_split} ({100*n_split/len(subset):.1f}%)")

    return records


# ── RUN ──────────────────────────────────────────────────────────────────────
records_g4  = run_diagnostic(G4_FILE,  "G4",  is_cc3=False)
records_cc3 = run_diagnostic(CC3_FILE, "CC3", is_cc3=True)


# ── PLOTS ─────────────────────────────────────────────────────────────────────

# Plot 1: Delta-alpha distribution for split_confirmed pairs
# Physics question: what is the angular scale of PandoraPFA splitting?
# Expected: peak at very small angles (< 0.05 rad)

fig, ax = plt.subplots(figsize=(7, 5))
bins_da = np.linspace(0, SPLIT_DA_MAX, 25)
for records, color, label, ls in [
    (records_g4,  "#2166ac", "G4",  "-"),
    (records_cc3, "#d6604d", "CC3", "--"),
]:
    da_vals = [r["best_da"] for r in records if r["cls"] == "split_confirmed"]
    ax.hist(da_vals, bins=bins_da, histtype="step",
            color=color, lw=2, linestyle=ls,
            label=f"{label} (N={len(da_vals)})")
ax.set_xlabel("Angular separation Δα between split fragments [rad]", fontsize=11)
ax.set_ylabel("Split-confirmed pairs / bin", fontsize=11)
ax.set_title("Angular scale of split-cluster pairs\n"
             "Signal B | E_reco < 10 GeV | split_confirmed only\n"
             "genStatus==1, E_mc≥10 GeV, not ISR", fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/split_delta_alpha_signalB_1k.png", dpi=150)
plt.close(fig)
print("\nSaved: split_delta_alpha_signalB_1k.png")


# Plot 2: Energy asymmetry scatter — fragment vs partner energy
# Physics question: how asymmetric is the split? Is fragment always soft?
# Expected: cluster of points near y-axis (fragment tiny, partner large)

fig, ax = plt.subplots(figsize=(7, 6))
for records, color, label in [
    (records_g4,  "#2166ac", "G4"),
    (records_cc3, "#d6604d", "CC3"),
]:
    split = [r for r in records if r["cls"] == "split_confirmed"]
    ax.scatter([r["e_reco"]      for r in split],
               [r["best_e_part"] for r in split],
               alpha=0.4, s=15, color=color, label=f"{label} (N={len(split)})")
ax.axvline(E_MC_MIN, color="black", lw=1, ls=":", alpha=0.5, label="E=10 GeV")
ax.set_xlabel("E_reco of fragment (Group B photon) [GeV]", fontsize=11)
ax.set_ylabel("E_reco of partner photon [GeV]", fontsize=11)
ax.set_title("Split-cluster energy asymmetry\n"
             "Signal B | split_confirmed | G4 vs CC3\n"
             "genStatus==1, E_mc≥10 GeV, not ISR", fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/split_energy_asymmetry_signalB_1k.png", dpi=150)
plt.close(fig)
print("Saved: split_energy_asymmetry_signalB_1k.png")


# Plot 3: Classification bar chart side by side
# Physics question: is split fraction the same in G4 and CC3?

categories = ["split_confirmed", "split_close_badsum",
              "split_faraway_goodsum", "split_partial", "unexplained_no_partner"]
colors_bar = ["#2ca02c", "#98df8a", "#ffbb78", "#ff7f0e", "#d62728"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, records, title in [(axes[0], records_g4, "G4"), (axes[1], records_cc3, "CC3")]:
    total  = len(records)
    counts = Counter(r["cls"] for r in records)
    vals   = [100 * counts.get(c, 0) / max(total, 1) for c in categories]
    bars   = ax.bar(range(len(categories)), vals,
                    color=colors_bar, edgecolor="black", lw=0.7)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of Group B photons [%]")
    ax.set_title(f"{title} — Group B classification\n"
                 f"Signal B | E_reco < 10 GeV (N={total})", fontsize=10)
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, vals):
        if val > 2:
            ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                    f"{val:.0f}%", ha="center", fontsize=9)
fig.suptitle("Group B photon classification — split-cluster diagnostic\n"
             "genStatus==1, E_mc≥10 GeV, not ISR", fontsize=11)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/split_classification_signalB_1k.png", dpi=150)
plt.close(fig)
print("Saved: split_classification_signalB_1k.png")

print(f"\nAll plots saved to: {PLOT_DIR}\nDone.")
