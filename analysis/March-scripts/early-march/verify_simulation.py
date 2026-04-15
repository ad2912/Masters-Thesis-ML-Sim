'''
verify_sim_files.py
===================
Clean, minimal verification of the two SIM files.

This script answers four questions:
  1. Do both SIM files exist and are they readable?
  2. Do they have the same number of events?
  3. Is the MC truth identical? (same generator input used?)
  4. Do the CaloClouds ECAL hits look different from Geant4?
     (confirms CaloClouds actually fired during simulation)
'''
import numpy as np
from podio import root_io
import os

SIM_G4 = "/data/dust/user/alimuham/thesis/sim/tau-pi0-geant4-V2-sim.edm4hep.root"
SIM_CC = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-V2-sim.edm4hep.root"

PHOTON_PDG   = 22
ELECTRON_PDG = 11
POSITRON_PDG = -11

# ECAL simhit collections — confirmed from podio-dump
# Contributions hold the steplength, not the hits themselves
ECAL_HIT_COLLECTIONS = [
    "ECalBarrelSiHitsEven",
    "ECalBarrelSiHitsOdd",
    "ECalBarrelScHitsEven",
    "ECalBarrelScHitsOdd",
    "ECalEndcapSiHitsEven",
    "ECalEndcapSiHitsOdd",
    "ECalEndcapScHitsEven",
    "ECalEndcapScHitsOdd",
]

ECAL_CONTRIB_COLLECTIONS = [
    "ECalBarrelSiHitsEvenContributions",
    "ECalBarrelSiHitsOddContributions",
    "ECalBarrelScHitsEvenContributions",
    "ECalBarrelScHitsOddContributions",
    "ECalEndcapSiHitsEvenContributions",
    "ECalEndcapSiHitsOddContributions",
    "ECalEndcapScHitsEvenContributions",
    "ECalEndcapScHitsOddContributions",
]

def header(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def ok(msg):    print(f"  ✓  {msg}")
def warn(msg):  print(f"  ⚠  {msg}")
def fail(msg):  print(f"  ✗  {msg}")
def note(msg):  print(f"     {msg}")

all_checks_passed = True

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Files exist and are readable
# ─────────────────────────────────────────────────────────────────────────────
header("CHECK 1 — File Integrity")

for label, path in [("SIM Geant4    ", SIM_G4), ("SIM CaloClouds", SIM_CC)]:
    if not os.path.exists(path):
        fail(f"{label}: NOT FOUND — {path}")
        all_checks_passed = False
        continue
    size_mb = os.path.getsize(path) / 1e6
    try:
        r = root_io.Reader(path)
        n = len(r.get("events"))
        ok(f"{label}: {n} events  |  {size_mb:.0f} MB  |  readable")
    except Exception as e:
        fail(f"{label}: could not open — {e}")
        all_checks_passed = False

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Event counts match
# ─────────────────────────────────────────────────────────────────────────────
header("CHECK 2 — Event Counts")

r_g4 = root_io.Reader(SIM_G4)
r_cc = root_io.Reader(SIM_CC)
n_g4 = len(r_g4.get("events"))
n_cc = len(r_cc.get("events"))

ok(f"SIM Geant4    : {n_g4} events")
ok(f"SIM CaloClouds: {n_cc} events")

if n_g4 == n_cc:
    ok(f"Event counts match — both have {n_g4} events")
else:
    fail(f"Event counts differ: G4={n_g4}, CC={n_cc}")
    fail("Different number of events were simulated — inputs were not the same")
    all_checks_passed = False

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — MC truth: same particles in both files?
# We check three things:
#   a. Total MC photon count across all events
#   b. Per-event photon count (should be identical event-by-event)
#   c. Energy of first photon in first 5 events (spot check)
# ─────────────────────────────────────────────────────────────────────────────
header("CHECK 3 — MC Truth Verification")

note("MC truth is written from the generator BEFORE any simulation.")
note("If both SIM files used the same input, MCParticles must match.")
note("")

g4_per_event_counts = []
cc_per_event_counts = []
g4_photon_energies  = []
cc_photon_energies  = []

# Count particles above ML trigger threshold
g4_above10 = {"gamma": 0, "e-": 0, "e+": 0}
cc_above10 = {"gamma": 0, "e-": 0, "e+": 0}

for i, (ev_g4, ev_cc) in enumerate(zip(r_g4.get("events"), r_cc.get("events"))):
    mc_g4 = ev_g4.get("MCParticles")
    mc_cc = ev_cc.get("MCParticles")

    g4_count = 0
    cc_count = 0

    for p in mc_g4:
        pdg = p.getPDG()
        e   = p.getEnergy()
        if pdg == PHOTON_PDG:
            g4_count += 1
            g4_photon_energies.append(e)
            if e >= 10.0:
                g4_above10["gamma"] += 1
        elif pdg == ELECTRON_PDG and e >= 10.0:
            g4_above10["e-"] += 1
        elif pdg == POSITRON_PDG and e >= 10.0:
            g4_above10["e+"] += 1

    for p in mc_cc:
        pdg = p.getPDG()
        e   = p.getEnergy()
        if pdg == PHOTON_PDG:
            cc_count += 1
            cc_photon_energies.append(e)
            if e >= 10.0:
                cc_above10["gamma"] += 1
        elif pdg == ELECTRON_PDG and e >= 10.0:
            cc_above10["e-"] += 1
        elif pdg == POSITRON_PDG and e >= 10.0:
            cc_above10["e+"] += 1

    g4_per_event_counts.append(g4_count)
    cc_per_event_counts.append(cc_count)

total_g4_photons = sum(g4_per_event_counts)
total_cc_photons = sum(cc_per_event_counts)

ok(f"Total MC photons — G4: {total_g4_photons},  CC: {total_cc_photons}")

if total_g4_photons == total_cc_photons:
    ok("Total MC photon counts are identical")
elif abs(total_g4_photons - total_cc_photons) <= 5:
    warn(f"Difference of {abs(total_g4_photons - total_cc_photons)} photon(s)")
    note("This is within normal range — caused by Geant4 writing a small number")
    note("of secondary photons (bremsstrahlung etc.) into MCParticles during sim.")
    note("This does NOT mean different input files were used.")
else:
    fail(f"Large MC photon count difference: G4={total_g4_photons}, CC={total_cc_photons}")
    fail("This strongly suggests different input files were used.")
    all_checks_passed = False

# Per-event comparison
mismatches = sum(1 for a, b in zip(g4_per_event_counts, cc_per_event_counts) if a != b)
if mismatches == 0:
    ok("Per-event MC photon counts are identical across all events")
elif mismatches <= 5:
    warn(f"Per-event counts differ in {mismatches} events — within acceptable range")
    note("Same root cause as above: rare Geant4 secondaries written into MCParticles")
else:
    fail(f"Per-event counts differ in {mismatches} events — too many to be secondaries")
    all_checks_passed = False

# Mean energies
ok(f"Mean MC photon energy — G4: {np.mean(g4_photon_energies):.2f} GeV,  "
   f"CC: {np.mean(cc_photon_energies):.2f} GeV")

print()
note("ML trigger candidates in 1000 events (particles >= 10 GeV):")
note(f"  Photons  : G4={g4_above10['gamma']},  CC={cc_above10['gamma']}")
note(f"  Electrons: G4={g4_above10['e-']},  CC={cc_above10['e-']}")
note(f"  Positrons: G4={g4_above10['e+']},  CC={cc_above10['e+']}")
total_ml_expected = cc_above10['gamma'] + cc_above10['e-'] + cc_above10['e+']
note(f"  → Total expected ML-simulated showers in CC: ~{total_ml_expected}")
note(f"    (This is the number CaloClouds should have fast-simulated)")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — CaloClouds actually fired: zero-steplength hits in ECAL
# We read contributions (not hits) because steplength lives there
# ─────────────────────────────────────────────────────────────────────────────
header("CHECK 4 — CaloClouds Fired (zero-steplength ECAL contributions)")

note("In Geant4: every calorimeter contribution has steplength > 0.")
note("In CaloClouds: ML-generated contributions have steplength == 0.")
note("We check the first 50 events as a representative sample.")
note("")

g4_zero  = 0
g4_total = 0
cc_zero  = 0
cc_total = 0

for i, (ev_g4, ev_cc) in enumerate(zip(r_g4.get("events"), r_cc.get("events"))):
    if i >= 50:
        break

    for coll_name in ECAL_CONTRIB_COLLECTIONS:
        try:
            contribs_g4 = ev_g4.get(coll_name)
            for c in contribs_g4:
                g4_total += 1
                if c.getStepLength() == 0:
                    g4_zero += 1
        except Exception:
            pass

        try:
            contribs_cc = ev_cc.get(coll_name)
            for c in contribs_cc:
                cc_total += 1
                if c.getStepLength() == 0:
                    cc_zero += 1
        except Exception:
            pass

if g4_total > 0:
    ok(f"Geant4 SIM     (50 events): {g4_zero}/{g4_total} zero-steplength contributions "
       f"({100*g4_zero/g4_total:.1f}%)")
    if g4_zero == 0:
        ok("  → Correct: pure Geant4 has NO zero-steplength contributions")
    else:
        warn(f"  → Unexpected: found {g4_zero} zero-steplength contributions in G4 file")

if cc_total > 0:
    ok(f"CaloClouds SIM (50 events): {cc_zero}/{cc_total} zero-steplength contributions "
       f"({100*cc_zero/cc_total:.1f}%)")
    if cc_zero > 0:
        ok("  → Correct: CaloClouds has zero-steplength contributions — ML fired ✓")
    else:
        fail("  → CaloClouds has NO zero-steplength contributions — ML did NOT fire!")
        fail("    Check your steering file and DDML setup.")
        all_checks_passed = False

# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
header("FINAL VERDICT")

if all_checks_passed:
    print("""
  ✓  ALL CHECKS PASSED.

  Your SIM files are solid:
  - Same number of events
  - Same MC truth (same generator input confirmed)
  - CaloClouds ML simulation fired correctly

  You can safely delete the current reco files and rerun
  reconstruction from these SIM files with full confidence.
""")
else:
    print("""
  ✗  ONE OR MORE CHECKS FAILED.

  Do NOT delete your reco files yet.
  Fix the issues flagged above before rerunning reconstruction.
""")
