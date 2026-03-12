"""
step1_inspect_generator.py
==========================
Inspect the filtered generator file BEFORE any simulation.

This is your ground truth. Everything you see in the SIM and RECO files
must be traceable back to what is in here.

Physics questions this answers:
  Q1. How many events passed your skim filter?
  Q2. In the first 1000 events (what you simulated): how many MC photons,
      electrons, and positrons are there above 10 GeV?
  Q3. What does the photon energy spectrum look like?
  Q4. How many photons per event on average?
  Q5. Are the events you care about (tau -> pi0 -> gamma) cleanly selected?

Run with:
    python3 step1_inspect_generator.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE
# ─────────────────────────────────────────────────────────────────────────────
GENERATOR_FILE = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/steering/tau_pi0_10GeV_filtered.edm4hep.root"
PLOT_DIR       = "/data/dust/user/alimuham/thesis/diagnostic_plots"
N_SIMULATED    = 1000   # how many events you passed to ddsim

os.makedirs(PLOT_DIR, exist_ok=True)

# PDG IDs
PHOTON   = 22
ELECTRON = 11
POSITRON = -11
TAU_PLUS  = -15
TAU_MINUS =  15
PI0       = 111

ML_TRIGGER_GEV = 10.0  # CaloClouds fires above this energy

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def ok(msg):   print(f"  ✓  {msg}")
def warn(msg): print(f"  ⚠  {msg}")
def note(msg): print(f"     {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# OPEN FILE
# ─────────────────────────────────────────────────────────────────────────────
header("Opening Generator File")

import podio
reader = podio.root_io.Reader(GENERATOR_FILE)
all_events = reader.get("events")
n_total = len(all_events)
ok(f"File opened: {GENERATOR_FILE}")
ok(f"Total events in filtered file: {n_total}")

if n_total < N_SIMULATED:
    warn(f"File has only {n_total} events, but you tried to simulate {N_SIMULATED}!")
    warn("ddsim would have only processed all {n_total} events, not 1000.")
    N_SIMULATED = n_total
else:
    ok(f"File has enough events for your 1000-event simulation run")


# ─────────────────────────────────────────────────────────────────────────────
# FULL FILE SCAN: counts across ALL events
# ─────────────────────────────────────────────────────────────────────────────
header("Full File Scan (all events)")

note("We scan ALL events so you know the full scope of your generator file.")
note("")

total_photons_all   = 0
total_electrons_all = 0
total_positrons_all = 0
total_photons_above10_all   = 0
total_electrons_above10_all = 0
total_positrons_above10_all = 0

for i, event in enumerate(reader.get("events")):
    mc = event.get("MCParticles")
    for p in mc:
        pdg = p.getPDG()
        e   = p.getEnergy()
        if pdg == PHOTON:
            total_photons_all += 1
            if e >= ML_TRIGGER_GEV:
                total_photons_above10_all += 1
        elif pdg == ELECTRON:
            total_electrons_all += 1
            if e >= ML_TRIGGER_GEV:
                total_electrons_above10_all += 1
        elif pdg == POSITRON:
            total_positrons_all += 1
            if e >= ML_TRIGGER_GEV:
                total_positrons_above10_all += 1

ok(f"Total MC photons   in full file : {total_photons_all}  "
   f"({total_photons_above10_all} above {ML_TRIGGER_GEV} GeV)")
ok(f"Total MC electrons in full file : {total_electrons_all}  "
   f"({total_electrons_above10_all} above {ML_TRIGGER_GEV} GeV)")
ok(f"Total MC positrons in full file : {total_positrons_all}  "
   f"({total_positrons_above10_all} above {ML_TRIGGER_GEV} GeV)")

print()
note("KEY INSIGHT — CaloClouds triggers on e+, e-, and gamma above 10 GeV.")
note(f"So in your CaloClouds simulation, across the full file it would")
note(f"fast-simulate roughly:")
ml_triggers_all = total_photons_above10_all + total_electrons_above10_all + total_positrons_above10_all
note(f"  {ml_triggers_all} ML-simulated showers  "
     f"({total_photons_above10_all} γ + {total_electrons_above10_all} e- + {total_positrons_above10_all} e+)")


# ─────────────────────────────────────────────────────────────────────────────
# FIRST 1000 EVENTS: detailed scan (what your simulation actually saw)
# ─────────────────────────────────────────────────────────────────────────────
header(f"First {N_SIMULATED} Events — What Your Simulation Actually Processed")

note("These are the events ddsim consumed. Every photon here should appear")
note("in both your Geant4 and CaloClouds SIM files as MCParticles.")
note("")

# Per-event storage
per_event_photons_total   = []
per_event_photons_above10 = []
per_event_electrons_above10 = []
per_event_positrons_above10 = []
per_event_ml_triggers     = []
per_event_tau_count       = []
per_event_pi0_count       = []

# Flat lists for histograms
all_photon_energies   = []
all_electron_energies = []
all_positron_energies = []

for i, event in enumerate(reader.get("events")):
    if i >= N_SIMULATED:
        break

    mc = event.get("MCParticles")

    ph_total = 0
    ph_above10 = 0
    el_above10 = 0
    po_above10 = 0
    n_tau = 0
    n_pi0 = 0

    for p in mc:
        pdg = p.getPDG()
        e   = p.getEnergy()

        if pdg == PHOTON:
            ph_total += 1
            all_photon_energies.append(e)
            if e >= ML_TRIGGER_GEV:
                ph_above10 += 1

        elif pdg == ELECTRON:
            all_electron_energies.append(e)
            if e >= ML_TRIGGER_GEV:
                el_above10 += 1

        elif pdg == POSITRON:
            all_positron_energies.append(e)
            if e >= ML_TRIGGER_GEV:
                po_above10 += 1

        elif abs(pdg) == TAU_MINUS:
            n_tau += 1

        elif pdg == PI0:
            n_pi0 += 1

    per_event_photons_total.append(ph_total)
    per_event_photons_above10.append(ph_above10)
    per_event_electrons_above10.append(el_above10)
    per_event_positrons_above10.append(po_above10)
    per_event_ml_triggers.append(ph_above10 + el_above10 + po_above10)
    per_event_tau_count.append(n_tau)
    per_event_pi0_count.append(n_pi0)


# Summary numbers
tot_ph   = sum(per_event_photons_total)
tot_ph10 = sum(per_event_photons_above10)
tot_el10 = sum(per_event_electrons_above10)
tot_po10 = sum(per_event_positrons_above10)
tot_ml   = sum(per_event_ml_triggers)
tot_tau  = sum(per_event_tau_count)
tot_pi0  = sum(per_event_pi0_count)

ok(f"Events scanned                  : {N_SIMULATED}")
ok(f"Total MC photons                : {tot_ph}  (mean {tot_ph/N_SIMULATED:.2f}/event)")
ok(f"MC photons >= 10 GeV            : {tot_ph10}  (mean {tot_ph10/N_SIMULATED:.2f}/event)")
ok(f"MC electrons >= 10 GeV          : {tot_el10}  (mean {tot_el10/N_SIMULATED:.2f}/event)")
ok(f"MC positrons >= 10 GeV          : {tot_po10}  (mean {tot_po10/N_SIMULATED:.2f}/event)")
print()
ok(f"TOTAL ML trigger particles      : {tot_ml}  (mean {tot_ml/N_SIMULATED:.2f}/event)")
note("  ↑ This is how many fast-simulated showers CaloClouds produced.")
note(f"  In your Geant4 reco file you found ~3000 'G4-type' photons.")
note(f"  In your CaloClouds reco file you found ~3000 G4-type + ~1000 ML-type photons.")
note(f"  The ML-type number ({tot_ph10} photon triggers) should match the ~1000 ML photons.")
print()
ok(f"Total taus in {N_SIMULATED} events     : {tot_tau}  (mean {tot_tau/N_SIMULATED:.2f}/event)")
ok(f"Total pi0s in {N_SIMULATED} events     : {tot_pi0}  (mean {tot_pi0/N_SIMULATED:.2f}/event)")

# Critical cross-check
print()
note("CROSS-CHECK TABLE:")
note(f"  Reco G4 file photon count  ≈ 3000   ←→  generator has {tot_ph} total MC photons")
note(f"  Reco CC ML photon count    ≈ 1000   ←→  generator has {tot_ph10} photons ≥10 GeV")
note(f"  Reco CC G4 photon count    ≈ 3000   ←→  these are the photons CaloClouds did NOT touch")
note(f"                                          (photons < 10 GeV, still Geant4-simulated)")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
header("Saving Diagnostic Plots")

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(f"Generator File Inspection — First {N_SIMULATED} Events\n"
             f"({GENERATOR_FILE.split('/')[-1]})", fontsize=13, fontweight='bold')

# ── Plot 1: Photon energy spectrum (all photons, log scale)
ax = axes[0, 0]
bins = np.linspace(0, 130, 80)
ax.hist(all_photon_energies, bins=bins, color="steelblue", edgecolor="navy", alpha=0.8)
ax.axvline(ML_TRIGGER_GEV, color="red", linestyle="--", linewidth=2,
           label=f"ML trigger = {ML_TRIGGER_GEV} GeV")
ax.set_xlabel("MC Photon Energy [GeV]", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("All MC Photon Energies\n(log y)", fontsize=10)
ax.set_yscale("log")
ax.legend(fontsize=9)

# ── Plot 2: Photons per event distribution
ax = axes[0, 1]
max_ph = max(per_event_photons_total)
ax.hist(per_event_photons_total, bins=range(0, max_ph + 2),
        color="steelblue", edgecolor="navy", alpha=0.8)
ax.set_xlabel("MC Photons per Event (all energies)", fontsize=11)
ax.set_ylabel("Number of Events", fontsize=11)
ax.set_title(f"Photons per Event\nmean={np.mean(per_event_photons_total):.1f}", fontsize=10)

# ── Plot 3: Photons >= 10 GeV per event
ax = axes[0, 2]
max_ph10 = max(per_event_photons_above10)
ax.hist(per_event_photons_above10, bins=range(0, max_ph10 + 2),
        color="tomato", edgecolor="darkred", alpha=0.8)
ax.set_xlabel(f"MC Photons ≥{ML_TRIGGER_GEV} GeV per Event", fontsize=11)
ax.set_ylabel("Number of Events", fontsize=11)
ax.set_title(f"High-E Photons per Event (ML-triggering)\nmean={np.mean(per_event_photons_above10):.2f}", fontsize=10)

# ── Plot 4: ML trigger particles per event (gamma + e+ + e-)
ax = axes[1, 0]
max_ml = max(per_event_ml_triggers)
ax.hist(per_event_ml_triggers, bins=range(0, max_ml + 2),
        color="darkorange", edgecolor="saddlebrown", alpha=0.8)
ax.set_xlabel("Total ML-trigger particles per event\n(γ + e⁻ + e⁺ ≥ 10 GeV)", fontsize=11)
ax.set_ylabel("Number of Events", fontsize=11)
ax.set_title(f"Total CaloClouds triggers/event\nmean={np.mean(per_event_ml_triggers):.2f}", fontsize=10)

# ── Plot 5: Electron and positron energy spectra
ax = axes[1, 1]
bins_lep = np.linspace(0, 130, 60)
ax.hist(all_electron_energies, bins=bins_lep, histtype="step",
        color="green", linewidth=2, label=f"e⁻ (N={len(all_electron_energies)})")
ax.hist(all_positron_energies, bins=bins_lep, histtype="step",
        color="purple", linewidth=2, label=f"e⁺ (N={len(all_positron_energies)})",
        linestyle="--")
ax.axvline(ML_TRIGGER_GEV, color="red", linestyle="--", linewidth=1.5,
           label=f"ML trigger = {ML_TRIGGER_GEV} GeV")
ax.set_xlabel("Energy [GeV]", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Electron & Positron Energies\n(CaloClouds also triggers on these!)", fontsize=10)
ax.set_yscale("log")
ax.legend(fontsize=9)

# ── Plot 6: Pi0 and tau counts per event
ax = axes[1, 2]
ax.hist(per_event_pi0_count, bins=range(0, max(per_event_pi0_count) + 2),
        color="gold", edgecolor="goldenrod", alpha=0.8, label="π⁰ per event")
ax.set_xlabel("Count per Event", fontsize=11)
ax.set_ylabel("Number of Events", fontsize=11)
ax.set_title(f"π⁰ per event\n(confirms tau→π⁰ decay filter worked)", fontsize=10)
ax.legend()

plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/step1_generator_inspection.png", dpi=150)
plt.close()
ok(f"Plot saved: {PLOT_DIR}/step1_generator_inspection.png")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL INTERPRETATION GUIDE
# ─────────────────────────────────────────────────────────────────────────────
header("How to Interpret These Numbers")

print("""
  WHAT THIS TELLS YOU ABOUT YOUR 3k vs 4k MYSTERY:
  ─────────────────────────────────────────────────

  Your Geant4 reco file has ~3000 photons.
  Your CaloClouds reco file has ~3000 + ~1000 = ~4000 photons.

  This is NOT a bug. Here is why:

  In the Geant4 simulation:
    → ALL particles are stepped through the full detector geometry.
    → Every photon from tau→pi0→gamma decay leaves Geant4-style hits.
    → PandoraPFA reconstructs these into ~3000 reco photons.

  In the CaloClouds simulation:
    → Particles ABOVE 10 GeV (gamma, e-, e+) are fast-simulated by the ML model.
    → Their calorimeter hits have steplength == 0 (ML-generated point cloud).
    → Particles BELOW 10 GeV are still handled by Geant4 normally.
    → PandoraPFA sees BOTH types of hits and reconstructs ~4000 photons.

  The extra ~1000 photons in CaloClouds are the ML-fast-simulated showers
  being reconstructed. The question is: are these genuinely EXTRA photons
  (double-counting), or are they REPLACING the Geant4 photons?

  You would expect: CaloClouds photons ≈ (G4 photons below 10 GeV) + (ML showers above 10 GeV)
  If that adds up to roughly 4000, your simulation is behaving correctly.

  NEXT CHECK: Compare this number with what the generator file shows:
    → Total MC photons in 1000 events                : printed above
    → MC photons >= 10 GeV (ML-simulated)            : printed above
    → MC photons < 10 GeV (still Geant4 in CC sim)   : difference of the two
""")
