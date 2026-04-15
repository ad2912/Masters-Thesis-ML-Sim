"""
validate_counts.py
==================
Fast event/photon count validation across GEN, G4, and CC3 files.

Metrics printed (per file, up to MAX_EVENTS):
  1. Total events in file
  2. Total MC photons
  3. genStatus == 1 photons
  4. genStatus == 1, E >= 10 GeV, direct pi0 daughter  (signal, strict)
  5. genStatus == 1, E >= 10 GeV, NOT direct e+/e- child  (signal, ISR-rejected)
  6. isHandledByFastSim == True  (CC3 only)

Speed notes:
  - pi0-daughter check uses a per-event index set (O(1) lookup), NOT recursive tree walk
  - No steplength / ECal hit scanning
  - ~1-2 min for 100k events

Run:
    source ~/source.sh
    python3 validate_counts.py
"""

import os
from podio import root_io

# ─── Paths ───────────────────────────────────────────────────────────────────
GEN = "/data/dust/user/alimuham/thesis/InputFiles/tau_pi0_10GeV_filtered_500kevents.edm4hep.root"
G4  = "/data/dust/user/alimuham/thesis/sim/tau-pi0-geant4-100kevents-sim.edm4hep.root"
CC3 = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-100kevents-sim.edm4hep.root"

MAX_EVENTS  = 100_000
E_THRESH    = 10.0   # GeV
PHOTON_PDG  = 22
PI0_PDG     = 111


def get_stats(label, path, is_cc3=False):
    if not os.path.exists(path):
        print(f"  ERROR: not found — {path}")
        return None

    print(f"  Reading {label} ...")

    reader    = root_io.Reader(path)
    all_events = reader.get("events")

    n_events       = 0
    n_photons      = 0
    n_gen1         = 0
    n_pi0_sig      = 0   # gen1, E>=10, direct pi0 daughter
    n_isr_rej_sig  = 0   # gen1, E>=10, NOT direct e+/e- child
    n_fastsim      = 0   # CC3 only

    for event in all_events:
        if n_events >= MAX_EVENTS:
            break
        n_events += 1

        particles = list(event.get("MCParticles"))

        # Build pi0-daughter index set for this event — O(n_particles), done once
        pi0_child_indices = set()
        for p in particles:
            if abs(p.getPDG()) == PI0_PDG:
                for child in p.getDaughters():
                    pi0_child_indices.add(child.getObjectID().index)

        for p in particles:
            if p.getPDG() != PHOTON_PDG:
                continue

            n_photons += 1

            gen_stat = p.getGeneratorStatus()
            energy   = p.getEnergy()

            if gen_stat != 1:
                continue
            n_gen1 += 1

            if energy < E_THRESH:
                continue

            # Direct parent PDGs — one level up only
            parent_pdgs = {abs(par.getPDG()) for par in p.getParents()}
            is_isr      = 11 in parent_pdgs  # direct e+/e- parent

            idx = p.getObjectID().index
            if idx in pi0_child_indices:
                n_pi0_sig += 1

            if not is_isr:
                n_isr_rej_sig += 1

            if is_cc3 and p.isHandledByFastSim():
                n_fastsim += 1

    return {
        "label"       : label,
        "events"      : n_events,
        "photons"     : n_photons,
        "gen1"        : n_gen1,
        "pi0_sig"     : n_pi0_sig,
        "isr_rej_sig" : n_isr_rej_sig,
        "fastsim"     : n_fastsim,
        "is_cc3"      : is_cc3,
    }


# ─── Run ─────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  Validation: event and photon counts")
print(f"  MAX_EVENTS = {MAX_EVENTS:,}")
print("=" * 60)

s_gen = get_stats("GEN (filtered input)", GEN, is_cc3=False)
s_g4  = get_stats("G4  (full sim)",       G4,  is_cc3=False)
s_cc3 = get_stats("CC3 (fast sim)",       CC3, is_cc3=True)

print()

if not all([s_gen, s_g4, s_cc3]):
    print("One or more files failed. Check paths.")
else:
    W  = 38   # metric column width
    C  = 13   # value column width
    SEP = "-" * (W + 3 * C + 8)

    def row(label, key, note=""):
        vals = [s_gen[key], s_g4[key], s_cc3[key]]
        cc_val = str(vals[2]) if vals[2] != 0 or s_cc3["is_cc3"] else "-"
        note_str = f"  ← {note}" if note else ""
        print(f"  {label:<{W}} | {str(vals[0]):<{C}} | {str(vals[1]):<{C}} | {cc_val:<{C}}{note_str}")

    print(f"  {'Metric':<{W}} | {'GEN':<{C}} | {'G4':<{C}} | {'CC3':<{C}}")
    print(SEP)

    row("1. Events in file (up to 100k)",      "events")
    print(SEP)
    row("2. Total MC photons",                 "photons")
    row("3. genStatus == 1",                   "gen1")
    print(SEP)
    row("4. gen1 + E>=10GeV + pi0 daughter",   "pi0_sig",
        "direct pi0 child only (1 level)")
    row("5. gen1 + E>=10GeV + not ISR",        "isr_rej_sig",
        "excludes direct e+/e- children")
    print(SEP)
    row("6. isHandledByFastSim (CC3 only)",    "fastsim")
    print(SEP)

    # Derived: fast sim fraction of signal
    sig  = s_cc3["isr_rej_sig"]
    fs   = s_cc3["fastsim"]
    frac = fs / sig if sig > 0 else 0.0
    print(f"\n  Fast-sim fraction (row 6 / row 5 in CC3): "
          f"{fs} / {sig} = {frac:.3f}  ({frac*100:.1f}%)")
    print()
    print("  Notes:")
    print("    Row 4: pi0 daughter = direct child of a PDG==111 particle (1-level, fast lookup)")
    print("    Row 5: ISR-rejected = photon whose direct parent is NOT e+ or e-")
    print("    Row 6: requires nightlies build (isHandledByFastSim flag)")
    print()
