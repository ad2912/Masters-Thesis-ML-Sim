"""
photon_analysis.py
==================
Compare photon PFOs between Geant4 and CaloClouds3 reconstructed events.

Physics question this script answers:
  Does CaloClouds3 reproduce the same photon energy spectrum, photon
  multiplicity per event, and total ECAL energy as Geant4?

Expected inputs:
  - tau_geant4_REC.edm4hep.root   (Geant4 full simulation, reconstructed)
  - tau_caloclouds_REC.edm4hep.root (CaloClouds3 ML sim, reconstructed)

Expected outputs:
  - plots/photon_energy.png         -- per-photon energy distribution
  - plots/photon_multiplicity.png   -- number of photon PFOs per event
  - plots/total_ecal_energy.png     -- sum of all photon energies per event
  - plots/photon_energy_vs_true.png -- reco photon energy vs true MC photon energy

Run from thesis-ml-sim/ root:
  python3 analysis/photon_analysis.py

Assumptions:
  - Photons are identified by PDG type == 22 in PandoraPFOs
  - RecoMCTruthLink maps PFOs to MCParticles for truth matching
  - Energy units in EDM4hep are GeV throughout
  - Both files have the same number of events (1000)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for cluster use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# podio/edm4hep Python bindings come with Key4hep environment
try:
    import podio
    from podio import root_io
    import edm4hep
except ImportError:
    print("ERROR: podio/edm4hep Python bindings not found.")
    print("Make sure you have sourced your Key4hep environment: source ~/source.sh")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECO_DIR  = os.path.join(BASE, "results", "reco")
PLOT_DIR  = os.path.join(BASE, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

G4_FILE  = os.path.join(RECO_DIR, "tau_geant4_REC.edm4hep.root")
CC3_FILE = os.path.join(RECO_DIR, "tau_caloclouds_REC.edm4hep.root")

for f in [G4_FILE, CC3_FILE]:
    if not os.path.exists(f):
        print(f"ERROR: File not found: {f}")
        print("Check that reconstruction has completed and paths are correct.")
        sys.exit(1)

# ── Styling ────────────────────────────────────────────────────────────────────
COLORS = {
    "geant4":     "#2166ac",   # blue
    "caloclouds": "#d6604d",   # red-orange
}
LABELS = {
    "geant4":     "Geant4 (full sim)",
    "caloclouds": "CaloClouds3 (ML fast sim)",
}

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "legend.fontsize":  11,
    "figure.dpi":       150,
})

# ── Data extraction ────────────────────────────────────────────────────────────

def extract_photon_data(filepath):
    """
    Loop over all events in a reconstructed EDM4hep file.
    For each event, collect:
      - energy of every photon PFO (PDG == 22)
      - number of photon PFOs in the event
      - total energy of all photon PFOs in the event
      - true MC photon energies (from MCParticles, PDG == 22, status==1)
      - matched reco/true energy pairs via RecoMCTruthLink

    Returns a dict of lists.
    """
    reader = root_io.Reader(filepath)

    photon_energies      = []   # energy of each photon PFO (all events flattened)
    photon_mult          = []   # number of photon PFOs per event
    total_ecal_energy    = []   # sum of photon PFO energies per event
    true_photon_energies = []   # true MC photon energies (flattened)
    matched_reco         = []   # reco energy for truth-matched photon PFOs
    matched_true         = []   # corresponding true energy

    n_events = 0

    for event in reader.get("events"):
        n_events += 1

        # ── PandoraPFOs: reconstructed particles ──────────────────────────────
        pfos = event.get("PandoraPFOs")

        event_photon_energies = []
        for pfo in pfos:
            if pfo.getPDG() == 22:                    # photon PDG code
                e = pfo.getEnergy()                   # GeV
                event_photon_energies.append(e)
                photon_energies.append(e)

        photon_mult.append(len(event_photon_energies))
        total_ecal_energy.append(sum(event_photon_energies))

        # ── MCParticles: true photons ─────────────────────────────────────────
        # Status == 1 means a stable final-state particle in the generator record
        mc_particles = event.get("MCParticles")
        for mc in mc_particles:
            if mc.getPDG() == 22:
                p = mc.getMomentum()
                e_true = mc.getEnergy()               # GeV
                true_photon_energies.append(e_true)

        # ── RecoMCTruthLink: match reco photon PFOs to true MC photons ────────
        # This link tells us: for each reco PFO, which MC particle caused it?
        # We use it to compare reco energy vs true energy for photons.
        try:
            links = event.get("RecoMCTruthLink")
            for link in links:
                reco = link.getFrom()   # the reconstructed PFO
                mc   = link.getTo()     # the true MC particle
                if reco.getPDG() == 22 and mc.getPDG() == 22:
                    matched_reco.append(reco.getEnergy())
                    matched_true.append(mc.getEnergy())
        except Exception:
            pass  # link may be empty in some events

    print(f"  Processed {n_events} events from {os.path.basename(filepath)}")
    print(f"  Total photon PFOs found: {len(photon_energies)}")
    print(f"  Mean photons per event:  {np.mean(photon_mult):.2f}")
    print(f"  Mean total ECAL energy:  {np.mean(total_ecal_energy):.2f} GeV")

    return {
        "photon_energies":      np.array(photon_energies),
        "photon_mult":          np.array(photon_mult),
        "total_ecal_energy":    np.array(total_ecal_energy),
        "true_photon_energies": np.array(true_photon_energies),
        "matched_reco":         np.array(matched_reco),
        "matched_true":         np.array(matched_true),
    }


# ── Run extraction ─────────────────────────────────────────────────────────────
print("\n── Extracting Geant4 data ──")
g4  = extract_photon_data(G4_FILE)

print("\n── Extracting CaloClouds3 data ──")
cc3 = extract_photon_data(CC3_FILE)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def compare_histogram(ax, data_g4, data_cc3, xlabel, title,
                       bins=40, xrange=None, density=True, log=False):
    """
    Draw two normalised histograms on the same axes for direct shape comparison.

    Normalised (density=True) so that differences in statistics don't obscure
    shape differences — we care about the shape, not the raw counts.
    """
    kwargs = dict(histtype="step", linewidth=2, density=density)
    rng = xrange if xrange else (
        min(data_g4.min() if len(data_g4) else 0,
            data_cc3.min() if len(data_cc3) else 0),
        max(data_g4.max() if len(data_g4) else 1,
            data_cc3.max() if len(data_cc3) else 1),
    )

    ax.hist(data_g4,  bins=bins, range=rng, color=COLORS["geant4"],
            label=LABELS["geant4"],  **kwargs)
    ax.hist(data_cc3, bins=bins, range=rng, color=COLORS["caloclouds"],
            label=LABELS["caloclouds"], **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalised entries" if density else "Entries")
    ax.set_title(title)
    ax.legend()
    if log:
        ax.set_yscale("log")

    # Print basic statistics as annotation
    if len(data_g4) and len(data_cc3):
        stats_text = (
            f"G4:  mean={data_g4.mean():.2f}, std={data_g4.std():.2f}\n"
            f"CC3: mean={data_cc3.mean():.2f}, std={data_cc3.std():.2f}"
        )
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))


# ── Plot 1: Per-photon energy distribution ─────────────────────────────────────
# Physics question: does CaloClouds reproduce the right photon energy spectrum?
# Expected outcome: both should show a broad distribution reflecting the range
# of photon energies from tau → π⁰ → γγ across 10-100 GeV tau energies.
# Differences indicate energy scale or resolution mismatch in CC3.

fig, ax = plt.subplots(figsize=(7, 5))
compare_histogram(
    ax,
    g4["photon_energies"],
    cc3["photon_energies"],
    xlabel="Photon PFO energy [GeV]",
    title="Reconstructed photon energy\n(all photon PFOs, all events)",
    bins=50,
    xrange=(0, 60),
)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "photon_energy.png")
fig.savefig(out)
print(f"\nSaved: {out}")
plt.close(fig)


# ── Plot 2: Photon multiplicity per event ──────────────────────────────────────
# Physics question: does CaloClouds cause PandoraPFA to find more or fewer
# photons per event than Geant4?
# Expected outcome: tau hadronic decay typically produces 1 π⁰ → 2 photons,
# so we expect a distribution peaking near 2 but with spread from other modes.
# If CC3 showers are wider, PandoraPFA may split them → higher multiplicity.
# If CC3 showers are narrower or lower energy, some may be missed → lower mult.

max_mult = max(
    g4["photon_mult"].max() if len(g4["photon_mult"]) else 0,
    cc3["photon_mult"].max() if len(cc3["photon_mult"]) else 0,
) + 1

fig, ax = plt.subplots(figsize=(7, 5))
bins_int = np.arange(0, max_mult + 1) - 0.5
ax.hist(g4["photon_mult"],  bins=bins_int, color=COLORS["geant4"],
        label=LABELS["geant4"],  histtype="step", linewidth=2, density=True)
ax.hist(cc3["photon_mult"], bins=bins_int, color=COLORS["caloclouds"],
        label=LABELS["caloclouds"], histtype="step", linewidth=2, density=True)
ax.set_xlabel("Number of photon PFOs per event")
ax.set_ylabel("Normalised entries")
ax.set_title("Photon PFO multiplicity per event")
ax.legend()
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
fig.tight_layout()
out = os.path.join(PLOT_DIR, "photon_multiplicity.png")
fig.savefig(out)
print(f"Saved: {out}")
plt.close(fig)


# ── Plot 3: Total EM energy per event ─────────────────────────────────────────
# Physics question: does CaloClouds get the total electromagnetic energy scale right?
# Expected outcome: sum of photon PFO energies per event should peak at a value
# reflecting the average EM fraction of tau decay energy.
# A shift in the mean between G4 and CC3 = energy scale bias in CaloClouds.

fig, ax = plt.subplots(figsize=(7, 5))
compare_histogram(
    ax,
    g4["total_ecal_energy"],
    cc3["total_ecal_energy"],
    xlabel="Total photon PFO energy per event [GeV]",
    title="Total reconstructed EM energy per event\n(sum of all photon PFO energies)",
    bins=40,
    xrange=(0, 80),
)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "total_ecal_energy.png")
fig.savefig(out)
print(f"Saved: {out}")
plt.close(fig)


# ── Plot 4: Reco vs True energy for truth-matched photons ─────────────────────
# Physics question: how well does each simulation reproduce photon energy?
# Expected outcome: points should lie along the diagonal (reco = true).
# Spread around the diagonal = energy resolution.
# Offset from diagonal = energy scale bias.
# This directly shows whether CC3 introduces a systematic bias or just
# different resolution compared to Geant4.

if len(g4["matched_true"]) > 0 and len(cc3["matched_true"]) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, key in zip(axes, [g4, cc3], ["geant4", "caloclouds"]):
        ax.scatter(data["matched_true"], data["matched_reco"],
                   alpha=0.4, s=8, color=COLORS[key])
        lim = max(data["matched_true"].max(), data["matched_reco"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Reco = True")
        ax.set_xlabel("True photon energy [GeV]")
        ax.set_ylabel("Reconstructed photon energy [GeV]")
        ax.set_title(f"Reco vs True photon energy\n{LABELS[key]}")
        ax.legend(fontsize=9)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "photon_energy_vs_true.png")
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)
else:
    print("Skipping reco-vs-true plot: no truth-matched photon pairs found.")
    print("This may happen with only 3 events — re-run on the full 1000-event files.")


# ── Summary printout ───────────────────────────────────────────────────────────
print("\n══ Summary ══════════════════════════════════════════")
print(f"{'Quantity':<35} {'Geant4':>12} {'CaloClouds3':>14}")
print("─" * 63)

def safe_mean(arr): return f"{arr.mean():.3f}" if len(arr) else "N/A"
def safe_std(arr):  return f"{arr.std():.3f}"  if len(arr) else "N/A"

rows = [
    ("Mean photon energy [GeV]",       g4["photon_energies"],   cc3["photon_energies"]),
    ("Std  photon energy [GeV]",       g4["photon_energies"],   cc3["photon_energies"]),
    ("Mean photons/event",             g4["photon_mult"],        cc3["photon_mult"]),
    ("Mean total EM energy/event [GeV]", g4["total_ecal_energy"], cc3["total_ecal_energy"]),
]
for label, dg4, dcc3 in rows:
    fn = safe_std if "Std" in label else safe_mean
    print(f"{label:<35} {fn(dg4):>12} {fn(dcc3):>14}")

print("═" * 63)
print("\nAll plots saved to:", PLOT_DIR)
