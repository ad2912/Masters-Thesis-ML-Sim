"""
tau_parentage_check_v3.py
==========================
Fix: skip generator placeholder copies by checking immediate-daughter
tau-ness BEFORE recursing (not after, which never fires since recursion
resolves past self-copies to the same real final state, double-counting
every physical tau).
"""

import numpy as np
from podio import root_io
from collections import Counter

REC_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"

TAU_PDG = 15
CHARGED_DECAY_PDGS = {211, -211, 321, -321, 13, -13, 11, -11}


def get_final_state_daughters(particle, visited=None):
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
n_true_taus_per_event = []
prong_counts = []
decay_lengths = []
event_has_buildupvertex = []
event_has_3prong_flags = []

for event in reader.get("events"):
    n_events += 1

    buv = event.get("BuildUpVertex")
    event_has_buildupvertex.append(len(buv) >= 1)

    all_taus = [p for p in event.get("MCParticles") if abs(p.getPDG()) == TAU_PDG]

    n_true_this_event = 0
    has_3prong_this_event = False

    for p in all_taus:
        immediate_pdgs = [d.getPDG() for d in p.getDaughters()]

        if any(abs(dp) == TAU_PDG for dp in immediate_pdgs):
            continue  # placeholder copy — the real decay entry is a separate tau in this list

        daughters = get_final_state_daughters(p)
        if len(daughters) == 0:
            continue

        n_true_this_event += 1
        dau_pdgs = [d.getPDG() for d in daughters]
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

print(f"Total events processed: {n_events}\n")

print("--- Corrected tau count per event ---")
print("Mean true taus/event:", n_true_taus_per_event.mean(), "(expect 2.00)")
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
print(f"\n3-prong fraction: {n_3prong}/{n_total_taus} = {100*n_3prong/n_total_taus:.2f}%")
print("(paper's reference value: 15.3% of all tau decays)")

print("\n--- Decay length [mm], corrected ---")
print(f"mean={decay_lengths.mean():.4f}  median={np.median(decay_lengths):.4f}")
print(f"5th pct={np.percentile(decay_lengths,5):.4f}  95th pct={np.percentile(decay_lengths,95):.4f}")

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
    print(f"Fraction of 3-prong-tau events that ALSO have BuildUpVertex: {100*n_overlap/n_3prong_events:.1f}%")
if n_buv_events > 0:
    print(f"Fraction of BuildUpVertex events that ARE 3-prong-tau events: {100*n_overlap/n_buv_events:.1f}%")
