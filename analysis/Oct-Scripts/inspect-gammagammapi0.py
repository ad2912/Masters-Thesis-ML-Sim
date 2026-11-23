# inspect_pi0_collection.py
# usage: python inspect_pi0_collection.py [N_EVENTS]
import sys, math
import numpy as np
from podio import root_io

N_EV = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
R = "/data/dust/user/alimuham/thesis/reco/"
FILES = {
    "G4":  R + "tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root",
    "CC3": R + "tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root",
}
COLL, PFO = "GammaGammaCandidatePi0s", "PandoraPFOs"

def p4(p):
    m = p.getMomentum()
    return np.array([p.getEnergy(), m.x, m.y, m.z])

def minv(v):
    m2 = v[0]**2 - v[1]**2 - v[2]**2 - v[3]**2
    return math.copysign(math.sqrt(abs(m2)), m2)

def oid(p):
    o = p.getObjectID()
    return (o.collectionID, o.index)

def inspect(label, path):
    print(f"\n{'='*20} {label} {'='*20}")
    reader = root_io.Reader(path)
    events = reader.get("events")
    first = events[0]
    print([n for n in first.getAvailableCollections() if "Gamma" in n or "Pandora" in n])
    n_ev = n_cand = n_ev_with = 0
    cands_per_ev, n_dau = [], []
    m_stored, m_dau, m_diff, E_cand, E_dau = [], [], [], [], []
    dau_mass, dau_pdg, in_pfo, p4_resid = [], [], [], []
    photon_use = {}
    printed = 0

    for ev in events:
        if n_ev >= N_EV:
            break
        n_ev += 1
        try:
            cands = ev.get(COLL)
        except Exception as e:
            print("could not read", COLL, "in event", n_ev, "->", e); break
            if cands is None:
                print(COLL, "returned None in event", n_ev); break
        pfo_ids = {oid(p) for p in ev.get(PFO)}
        cands_per_ev.append(len(cands))
        n_ev_with += len(cands) > 0
        for c in cands:
            n_cand += 1
            daus = list(c.getParticles())
            n_dau.append(len(daus))
            m_stored.append(c.getMass())
            E_cand.append(c.getEnergy())
            if len(daus) != 2:
                continue
            v = sum(p4(d) for d in daus)
            mm = minv(v)
            m_dau.append(mm)
            m_diff.append(mm - c.getMass())
            E_dau.append(v[0])
            p4_resid.append(np.abs(v - p4(c)).max())
            for d in daus:
                dau_mass.append(d.getMass())
                dau_pdg.append(d.getPDG())
                in_pfo.append(oid(d) in pfo_ids)
                k = (n_ev, oid(d))
                photon_use[k] = photon_use.get(k, 0) + 1
            if printed < 5:
                printed += 1
                print(f"ev{n_ev} cand: pdg={c.getPDG()} q={c.getCharge()} "
                      f"E={c.getEnergy():.2f} m_stored={c.getMass():.4f} "
                      f"m_from_daughters={mm:.4f} "
                      f"dauE={[round(d.getEnergy(),2) for d in daus]} "
                      f"dau_pdg={[d.getPDG() for d in daus]}")

    a = lambda x: np.array(x, dtype=float)
    print(f"events read: {n_ev}, events with >=1 candidate: {n_ev_with}, total candidates: {n_cand}")
    if n_cand == 0:
        return
    print("candidates/event (mean, max):", np.mean(cands_per_ev), max(cands_per_ev))
    print("n daughters per candidate (value:count):", dict(zip(*np.unique(n_dau, return_counts=True))))
    ms = a(m_stored)
    print(f"STORED mass: min={ms.min():.5f} max={ms.max():.5f} n_unique={len(np.unique(np.round(ms,6)))}")
    if m_dau:
        md = a(m_dau)
        print(f"DAUGHTER-SUM mass: mean={md.mean():.4f} std={md.std():.5f} "
              f"min={md.min():.4f} max={md.max():.4f}")
        print("  percentiles 5/25/50/75/95:", np.round(np.percentile(md, [5,25,50,75,95]), 4))
        print(f"  mass(dau) - stored mass: mean={a(m_diff).mean():.5f} std={a(m_diff).std():.5f}")
        print(f"  max |p4_candidate - sum(p4_daughters)| per cand: median={np.median(p4_resid):.2e} max={np.max(p4_resid):.2e}")
        print(f"  daughter PDGs: {dict(zip(*np.unique(dau_pdg, return_counts=True)))}")
        print(f"  daughter mass field: min={a(dau_mass).min():.2e} max={a(dau_mass).max():.2e}")
        print(f"  daughters found as objects in {PFO}: {100*np.mean(in_pfo):.1f}%")
        reuse = np.array(list(photon_use.values()))
        print(f"  photon reuse across candidates (uses:count): {dict(zip(*np.unique(reuse, return_counts=True)))}")

for lab, f in FILES.items():
    inspect(lab, f)
