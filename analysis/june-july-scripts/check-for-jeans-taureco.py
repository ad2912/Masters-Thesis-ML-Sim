"""
Full feasibility check for Jeans-style impact-parameter tau reconstruction.
Run on NAF where the files live:

    python3 tau_reco_feasibility_check.py

Paste the FULL printed output back for interpretation.
"""

import uproot
import numpy as np

REC_FILE = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"
SIM_FILE = "/data/dust/user/alimuham/thesis/sim/tau-pi0-caloclouds-100kevents-sim.edm4hep.root"

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------
section("0. Available keys (REC file) — vertex-related collections")
# ---------------------------------------------------------------
try:
    frec = uproot.open(REC_FILE)
    trec = frec["events"]
    vertex_colls = sorted(set(
        k.split("/")[0] for k in trec.keys()
        if k.split("/")[0].endswith("Vertex") or k.split("/")[0].endswith("Vertices")
    ))
    print("Vertex-like collections found:", vertex_colls)
except Exception as e:
    print("FAILED:", e)


# ---------------------------------------------------------------
section("1. Population + precision of ALL vertex collections")
# ---------------------------------------------------------------
vertex_candidates = [
    "PrimaryVertex", "BuildUpVertex", "BuildUpVertex_V0",
    "ProngVertices", "KinkVertices", "SplitVertices", "V0Vertices",
]
vertex_results = {}
for coll in vertex_candidates:
    try:
        xkey = f"{coll}/{coll}.position.x"
        if xkey not in trec.keys():
            print(f"{coll}: collection/branch not found, skipping")
            continue
        x = trec[xkey].array(library="np")
        sizes = np.array([len(v) for v in x])
        n_events_with = (sizes >= 1).sum()
        mean_per_event = sizes.mean()
        print(f"\n{coll}:")
        print(f"  events with >=1 vertex: {n_events_with} / {len(sizes)}  ({100*n_events_with/len(sizes):.2f}%)")
        print(f"  mean vertices/event: {mean_per_event:.4f}")
        print(f"  max vertices in one event: {sizes.max()}")

        covkey = f"{coll}/{coll}.covMatrix.values[6]"
        if covkey in trec.keys():
            cov = trec[covkey].array(library="np")
            flat_cov = np.concatenate([c for c in cov if len(c) > 0]) if n_events_with > 0 else np.array([])
            if len(flat_cov) > 0:
                sx = np.sqrt(np.abs(flat_cov[:, 0]))
                sy = np.sqrt(np.abs(flat_cov[:, 2]))
                sz = np.sqrt(np.abs(flat_cov[:, 5]))
                print(f"  sigma_x [um]: mean={sx.mean()*1000:.3f} median={np.median(sx)*1000:.3f}")
                print(f"  sigma_y [um]: mean={sy.mean()*1000:.3f} median={np.median(sy)*1000:.3f}")
                print(f"  sigma_z [um]: mean={sz.mean()*1000:.3f} median={np.median(sz)*1000:.3f}")

        npart = None
        pbkey = f"{coll}/{coll}.particles_begin"
        pekey = f"{coll}/{coll}.particles_end"
        if pbkey in trec.keys() and pekey in trec.keys():
            pb = np.concatenate(trec[pbkey].array(library="np")) if n_events_with > 0 else np.array([])
            pe = np.concatenate(trec[pekey].array(library="np")) if n_events_with > 0 else np.array([])
            if len(pb) > 0:
                npart = pe - pb
                print(f"  n_particles per vertex: {dict(zip(*np.unique(npart, return_counts=True)))}")

        vertex_results[coll] = dict(sizes=sizes, mean_per_event=mean_per_event)
    except Exception as e:
        print(f"{coll}: FAILED - {e}")


# ---------------------------------------------------------------
section("2. Fixing tau double-counting — generatorStatus breakdown")
# ---------------------------------------------------------------
try:
    pdg = trec["MCParticles/MCParticles.PDG"].array(library="np")
    gen = trec["MCParticles/MCParticles.generatorStatus"].array(library="np")
    db = trec["MCParticles/MCParticles.daughters_begin"].array(library="np")
    de = trec["MCParticles/MCParticles.daughters_end"].array(library="np")
    vx = trec["MCParticles/MCParticles.vertex.x"].array(library="np")
    vy = trec["MCParticles/MCParticles.vertex.y"].array(library="np")
    vz = trec["MCParticles/MCParticles.vertex.z"].array(library="np")
    epx = trec["MCParticles/MCParticles.endpoint.x"].array(library="np")
    epy = trec["MCParticles/MCParticles.endpoint.y"].array(library="np")
    epz = trec["MCParticles/MCParticles.endpoint.z"].array(library="np")

    all_tau_status = []
    for i in range(len(pdg)):
        mask = np.abs(pdg[i]) == 15
        all_tau_status.extend(gen[i][mask].tolist())
    all_tau_status = np.array(all_tau_status)
    print("ALL tau entries (any status) - breakdown:", dict(zip(*np.unique(all_tau_status, return_counts=True))))
    print("Total tau entries (any status):", len(all_tau_status))

    # "Physical decaying" tau = has daughters (real decay recorded), avoids doc/duplicate copies
    n_physical_taus_per_event = []
    decay_lengths = []
    n_charged_prongs_list = []
    prong_event_idx = []
    tau_event_idx = []

    for i in range(len(pdg)):
        is_tau = np.abs(pdg[i]) == 15
        has_daughters = de[i] > db[i]
        physical = is_tau & has_daughters
        idxs = np.where(physical)[0]
        n_physical_taus_per_event.append(len(idxs))
        for j in idxs:
            dx = epx[i][j] - vx[i][j]
            dy = epy[i][j] - vy[i][j]
            dz = epz[i][j] - vz[i][j]
            L = np.sqrt(dx**2 + dy**2 + dz**2)
            decay_lengths.append(L)
            tau_event_idx.append(i)

    n_physical_taus_per_event = np.array(n_physical_taus_per_event)
    decay_lengths = np.array(decay_lengths)
    print("\nUsing 'has_daughters' filter to identify physical (non-duplicate) tau decays:")
    print("mean physical taus/event:", n_physical_taus_per_event.mean())
    print("distribution of taus/event:", dict(zip(*np.unique(n_physical_taus_per_event, return_counts=True))))
    print("N physical tau decays found:", len(decay_lengths))
    print("decay length [mm]: mean=%.4f median=%.4f  5th=%.4f  95th=%.4f" % (
        decay_lengths.mean(), np.median(decay_lengths),
        np.percentile(decay_lengths, 5), np.percentile(decay_lengths, 95)))
except Exception as e:
    print("FAILED:", e)


# ---------------------------------------------------------------
section("3. Tau prong multiplicity (1-prong vs 3-prong vs leptonic)")
# ---------------------------------------------------------------
try:
    charge = trec["MCParticles/MCParticles.charge"].array(library="np")

    prong_counts = []
    decay_types = []
    event_of_tau = []
    for i in range(len(pdg)):
        is_tau = np.abs(pdg[i]) == 15
        has_daughters = de[i] > db[i]
        physical = is_tau & has_daughters
        idxs = np.where(physical)[0]
        for j in idxs:
            d_start, d_end = db[i][j], de[i][j]
            # daughters are indices into the same event's MCParticles array
            daughter_pdgs = pdg[i][d_start:d_end]
            daughter_charge = charge[i][d_start:d_end]
            daughter_gen = gen[i][d_start:d_end]
            stable_charged = (daughter_gen == 1) & (daughter_charge != 0)
            n_charged = stable_charged.sum()
            has_e = np.any(np.abs(daughter_pdgs) == 11)
            has_mu = np.any(np.abs(daughter_pdgs) == 13)
            if has_e or has_mu:
                dtype = "leptonic"
            elif n_charged == 1:
                dtype = "1-prong"
            elif n_charged == 3:
                dtype = "3-prong"
            else:
                dtype = f"other({n_charged}prong)"
            prong_counts.append(n_charged)
            decay_types.append(dtype)
            event_of_tau.append(i)

    decay_types = np.array(decay_types)
    prong_counts = np.array(prong_counts)
    event_of_tau = np.array(event_of_tau)
    print("Decay type breakdown:", dict(zip(*np.unique(decay_types, return_counts=True))))
except Exception as e:
    print("FAILED:", e)


# ---------------------------------------------------------------
section("4. Correlation: does PrimaryVertex quality depend on tau topology?")
# ---------------------------------------------------------------
try:
    pv_x = trec["PrimaryVertex/PrimaryVertex.position.x"].array(library="np")
    pv_pb = trec["PrimaryVertex/PrimaryVertex.particles_begin"].array(library="np")
    pv_pe = trec["PrimaryVertex/PrimaryVertex.particles_end"].array(library="np")
    n_events = len(pv_x)

    pv_nparticles_per_event = np.array([
        (pv_pe[i][0] - pv_pb[i][0]) if len(pv_pb[i]) > 0 else -1
        for i in range(n_events)
    ])

    good_pv_events = set(np.where(pv_nparticles_per_event >= 2)[0].tolist())
    bad_pv_events = set(np.where(pv_nparticles_per_event < 2)[0].tolist())

    for label, ev_set in [("PrimaryVertex GOOD (>=2 tracks)", good_pv_events),
                           ("PrimaryVertex DEFAULT (<2 tracks)", bad_pv_events)]:
        mask = np.isin(event_of_tau, list(ev_set))
        if mask.sum() > 0:
            sub_types = decay_types[mask]
            print(f"\n{label}  (n_events={len(ev_set)}, n_tau_entries={mask.sum()}):")
            print("  ", dict(zip(*np.unique(sub_types, return_counts=True))))
except Exception as e:
    print("FAILED:", e)


# ---------------------------------------------------------------
section("5. Do 3-prong tau events get a usable BuildUpVertex / ProngVertices instead?")
# ---------------------------------------------------------------
try:
    events_with_3prong = set(event_of_tau[decay_types == "3-prong"].tolist())
    print(f"Events containing a 3-prong tau decay: {len(events_with_3prong)}")

    for coll in ["BuildUpVertex", "ProngVertices", "KinkVertices"]:
        if coll not in vertex_results:
            print(f"{coll}: not available/not populated at all — see section 1")
            continue
        sizes = vertex_results[coll]["sizes"]
        pop_events = set(np.where(sizes >= 1)[0].tolist())
        overlap = events_with_3prong & pop_events
        print(f"{coll}: populated in {len(pop_events)} events total; "
              f"overlap with 3-prong-tau events: {len(overlap)} "
              f"({100*len(overlap)/max(len(events_with_3prong),1):.1f}% of 3-prong events)")
except Exception as e:
    print("FAILED:", e)


# ---------------------------------------------------------------
section("6. SIM file — VXD/tracker hit density check (100k events)")
# ---------------------------------------------------------------
try:
    fsim = uproot.open(SIM_FILE)
    tsim = fsim["events"]
    sim_keys = sorted(set(
        k.split("/")[0] for k in tsim.keys()
        if "VXD" in k or "SIT" in k or "FTD" in k or "TPC" in k
    ))
    print("Sim tracker-related collections:", sim_keys)

    for coll in ["VXDCollection", "SITCollection", "TPCCollection"]:
        key = f"{coll}/{coll}.cellID"
        alt_key = f"{coll}/{coll}.energy" if key not in tsim.keys() else key
        # just get hit counts per event via any always-present field
        found_key = None
        for cand in [f"{coll}/{coll}.cellID", f"{coll}/{coll}.time", f"{coll}/{coll}.position.x"]:
            if cand in tsim.keys():
                found_key = cand
                break
        if found_key is None:
            print(f"{coll}: no usable branch found for hit counting")
            continue
        arr = tsim[found_key].array(library="np", entry_stop=20000)  # subsample for speed
        sizes = np.array([len(v) for v in arr])
        print(f"{coll}: mean hits/event (first 20k events) = {sizes.mean():.2f}, "
              f"median = {np.median(sizes):.1f}, events with 0 hits = {(sizes==0).sum()}")
except Exception as e:
    print("FAILED (sim file check):", e)


# ---------------------------------------------------------------
section("7. SIM vs REC event-count sanity check")
# ---------------------------------------------------------------
try:
    print(f"REC file events: {trec.num_entries}")
    print(f"SIM file events: {tsim.num_entries}")
    print("NOTE: sim=100k, rec=10k -> REC is presumably a subset/subsample of SIM. "
          "If you need sim-level truth for the SAME events as in REC, confirm whether "
          "the first 10k sim events correspond to the REC file, or if REC used a "
          "different/shuffled subset.")
except Exception as e:
    print("FAILED:", e)


print("\n" + "=" * 70)
print("DONE. Paste full output back.")
print("=" * 70)
