
"""
tau_parentage_check_v2.py
==========================
Fixes two bugs found in the previous version:
  1. Immediate-daughter check missed resonance decay steps (rho, a1) -
     now recursively descends to generatorStatus==1 (stable) daughters.
  2. Diagnoses the bimodal 2-vs-4 tau count anomaly by printing
     generatorStatus values for tau entries in "4-tau" events, before
     trusting any final count.

Run with:
    source ~/source.sh
    python3 tau_parentage_check_v2.py
"""

import numpy as np
from podio import root_io
from collections import Counter

REC_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"

TAU_PDG = 15
CHARGED_DECAY_PDGS = {211, -211, 321, -321, 13, -13, 11, -11}


def get_final_state_daughters(particle, visited=None):
    """Recursively walk down to generatorStatus==1 (stable) daughters."""
    if visited is None:
        visited = set()

    result = []
    for d in particle.getDaughters():
        oid = d.getObjectID()
        key = (oid.collectionID, oid.index)
        if key in visited:
            continue
        visited.add(key)

        if d.getGeneratorStatus() == 1:
            result.append(d)
        else:
            result.extend(get_final_state_daughters(d, visited))
    return result


def decay_length(p):
    v = p.getVertex()
    ep = p.getEndpoint()
    dx, dy, dz = ep.x - v.x, ep.y - v.y, ep.z - v.z
    return (dx**2 + dy**2 + dz**2) ** 0.5


reader = root_io.Reader(REC_FILE)

n_events = 0

# --- Diagnostic pass: understand the 2-vs-4 tau count structure first ---
print("="*70)
print("DIAGNOSTIC: generatorStatus values for tau entries, first 5 "
      "'4-tau' events")
print("="*70)

diagnostic_events_shown = 0
for event in reader.get("events"):
    taus_this_event = [p for p in event.get("MCParticles")
                        if abs(p.getPDG()) == TAU_PDG]
    if len(taus_this_event) == 4 and diagnostic_events_shown < 5:
        diagnostic_events_shown += 1
        print(f"\nEvent with 4 tau entries (#{diagnostic_events_shown}):")
        for t in taus_this_event:
            n_dau = len(t.getDaughters())
            dau_pdgs = [d.getPDG() for d in t.getDaughters()]
            print(f"  PDG={t.getPDG():+d}  genStatus={t.getGeneratorStatus()}  "
                  f"nDaughters={n_dau}  daughterPDGs={dau_pdgs}")
    if diagnostic_events_shown >= 5:
        break

# --- Main pass: corrected counting with recursive descent ---
print("\n" + "="*70)
print("MAIN PASS: recursive descent to stable final-state daughters")
print("="*70)

reader = root_io.Reader(REC_FILE)  # re-open, generator was consumed above

n_events = 0
n_true_taus_per_event = []
n_taus_per_event_by_status = []  # for filtering experiments
prong_counts = []
decay_lengths = []
event_has_buildupvertex = []
event_has_3prong_flags = []

for event in reader.get("events"):
    n_events += 1

    buv = event.get("BuildUpVertex")
    has_buv = len(buv) >= 1
    event_has_buildupvertex.append(has_buv)

    n_true_this_event = 0
    has_3prong_this_event = False

    all_taus = [p for p in event.get("MCParticles") if abs(p.getPDG()) == TAU_PDG]
    status_counts = Counter(t.getGeneratorStatus() for t in all_taus)
    n_taus_per_event_by_status.append(status_counts)

    for p in all_taus:
        daughters = get_final_state_daughters(p)
        if len(daughters) == 0:
            continue

        dau_pdgs = [d.getPDG() for d in daughters]

        if any(abs(dp) == TAU_PDG for dp in dau_pdgs):
            continue  # safety check, shouldn't trigger at stable-particle depth

        n_true_this_event += 1
        n_charged = sum(1 for dp in dau_pdgs if abs(dp) in CHARGED_DECAY_PDGS)
        prong_counts.append(n_charged)
        decay_lengths.append(decay_length(p))

        if n_charged == 3:
            has_3prong_this_event = True

    n_true_taus_per_event.append(n_true_this_event)
    event_has_3prong_flags.append(has_3prong_this_event)

n_true_taus_per_event = np.array(n_true_taus_per_event)
prong_counts = np.array(prong_counts)
decay_lengths = np.array(decay_lengths)
event_has_buildupvertex = np.array(event_has_buildupvertex)
event_has_3prong_flags = np.array(event_has_3prong_flags)

print(f"\nTotal events processed: {n_events}")

print("\n--- generatorStatus breakdown across all tau entries (all events) ---")
combined_status = Counter()
for sc in n_taus_per_event_by_status:
    combined_status.update(sc)
for status, count in sorted(combined_status.items()):
    print(f"  status={status}: {count} tau entries")

print("\n--- Corrected tau count per event (recursive descent) ---")
print("Mean true taus/event:", n_true_taus_per_event.mean(), "(expect ~2.0)")
dist = Counter(n_true_taus_per_event.tolist())
for k in sorted(dist):
    print(f"  {k} taus: {dist[k]} events")

print("\n--- Prong distribution (all true, final tau decays) ---")
prong_dist = Counter(prong_counts.tolist())
n_total_taus = len(prong_counts)
for k in sorted(prong_dist):
    pct = 100 * prong_dist[k] / n_total_taus
    print(f"  {k}-prong: {prong_dist[k]}  ({pct:.2f}%)")

n_3prong = prong_dist.get(3, 0)
print(f"\n3-prong fraction: {n_3prong}/{n_total_taus} = "
      f"{100*n_3prong/n_total_taus:.2f}%")
print("(paper's reference value: 15.3% of all tau decays)")

print("\n--- Decay length [mm], corrected ---")
print(f"mean={decay_lengths.mean():.4f}  median={np.median(decay_lengths):.4f}")
print(f"5th pct={np.percentile(decay_lengths,5):.4f}  "
      f"95th pct={np.percentile(decay_lengths,95):.4f}")

print("\n" + "="*70)
print("BuildUpVertex overlap check")
print("="*70)
overlap = event_has_3prong_flags & event_has_buildupvertex
n_3prong_events = event_has_3prong_flags.sum()
n_buv_events = event_has_buildupvertex.sum()
n_overlap = overlap.sum()

print(f"Events with >=1 true 3-prong tau : {n_3prong_events}")
print(f"Events with BuildUpVertex populated: {n_buv_events}")
print(f"Overlap (both true): {n_overlap}")
if n_3prong_events > 0:
    print(f"Fraction of 3-prong-tau events that ALSO have BuildUpVertex: "
          f"{100*n_overlap/n_3prong_events:.1f}%")
if n_buv_events > 0:
    print(f"Fraction of BuildUpVertex events that ARE 3-prong-tau events: "
          f"{100*n_overlap/n_buv_events:.1f}%")

