# build_pi0_npz.py
# usage: python3 build_pi0_npz.py <REC file> <out.npz> -n 1000
import math, argparse
from collections import defaultdict
import numpy as np
import edm4hep
from podio import root_io

ap = argparse.ArgumentParser()
ap.add_argument("infile"); ap.add_argument("outfile")
ap.add_argument("-n", type=int, default=100000)
a = ap.parse_args()

COLL, LINK, PFO = "GammaGammaCandidatePi0s", "RecoMCTruthLink", "PandoraPFOs"
NAN = float("nan")

def key(p):
    o = p.getObjectID(); return (o.collectionID, o.index)

def decode(w):
    r = int(round(w)); return (r // 10000) / 1000.0, (r % 10000) / 1000.0

MC_KEYS = ["idx","pdg","gen","E","theta","phi","fast","r_end","z_end","n_dau","n_par",
           "par0_pdg","par0_idx","par1_pdg","pi0_par_idx","any_par_e"]
EMPTY = dict(idx=-1,pdg=0,gen=-1,E=NAN,theta=NAN,phi=NAN,fast=-1,r_end=NAN,z_end=NAN,
             n_dau=-1,n_par=-1,par0_pdg=0,par0_idx=-1,par1_pdg=0,pi0_par_idx=-1,any_par_e=-1)

def mc_info(m):
    p4 = edm4hep.utils.p4(m)
    ep = m.getEndpoint()
    has_ep = not (ep.x == 0 and ep.y == 0 and ep.z == 0)
    par = list(m.getParents())
    return dict(
        idx=m.getObjectID().index, pdg=m.getPDG(), gen=m.getGeneratorStatus(),
        E=m.getEnergy(),
        theta=math.degrees(float(p4.theta())), phi=math.degrees(float(p4.phi())),
        fast=int(m.isHandledByFastSim()),
        r_end=math.hypot(ep.x, ep.y) if has_ep else NAN,
        z_end=ep.z if has_ep else NAN,
        n_dau=len(list(m.getDaughters())), n_par=len(par),
        par0_pdg=par[0].getPDG() if len(par) > 0 else 0,
        par0_idx=par[0].getObjectID().index if len(par) > 0 else -1,
        par1_pdg=par[1].getPDG() if len(par) > 1 else 0,
        pi0_par_idx=next((q.getObjectID().index for q in par if q.getPDG() == 111), -1),
        any_par_e=int(any(abs(q.getPDG()) == 11 for q in par)),
    )

R = defaultdict(list)   # one row per candidate
V = defaultdict(list)   # one row per event (all events read)
def put(name, val): R[name].append(val)

events = root_io.Reader(a.infile).get("events")
n_ev = n_ev_cand = n_bad = 0

for ev in events:
    if n_ev >= a.n: break
    n_ev += 1
    if n_ev % 10000 == 0: print(f"  {n_ev} events", flush=True)
    hdr = ev.get("EventHeader")[0]
    cands = ev.get(COLL)

    n_g = n_g5 = 0
    for pf in ev.get(PFO):
        if pf.getPDG() == 22:
            n_g += 1
            m = pf.getMomentum()
            if math.sqrt(m.x**2 + m.y**2 + m.z**2) >= 0.5: n_g5 += 1
    V["ev_loop"].append(n_ev - 1); V["ev_run"].append(hdr.getRunNumber())
    V["ev_evt"].append(hdr.getEventNumber()); V["ev_n_cands"].append(len(cands))
    V["ev_n_gamma_pfo"].append(n_g); V["ev_n_gamma_pfo_p05"].append(n_g5)

    if len(cands) == 0: continue
    n_ev_cand += 1

    links = defaultdict(list)
    for l in ev.get(LINK):
        cw, tw = decode(l.getWeight())
        links[key(l.getFrom())].append((cw, tw, l.getTo()))

    for ci, c in enumerate(cands):
        daus = sorted(c.getParticles(), key=lambda d: -d.getEnergy())
        if len(daus) != 2:
            n_bad += 1; continue
        put("loop_evt", n_ev - 1); put("run", hdr.getRunNumber())
        put("evt", hdr.getEventNumber()); put("cand_idx", ci)
        cm = c.getMomentum()
        put("c_E", c.getEnergy()); put("c_px", cm.x); put("c_py", cm.y)
        put("c_pz", cm.z); put("c_mass_stored", c.getMass())
        for k, d in enumerate(daus):
            m = d.getMomentum(); pre = f"d{k}_"
            put(pre+"pfo_idx", d.getObjectID().index)
            put(pre+"E", d.getEnergy()); put(pre+"px", m.x)
            put(pre+"py", m.y); put(pre+"pz", m.z)
            put(pre+"mass_field", d.getMass())
            ls = sorted(links.get(key(d), []), key=lambda t: -t[0])
            put(pre+"n_links", len(ls))
            put(pre+"sum_cw", sum(t[0] for t in ls))
            for j in range(2):
                lp = f"{pre}l{j}_"
                if j < len(ls):
                    cw, tw, mc = ls[j]; info = mc_info(mc)
                else:
                    cw, tw, info = -1.0, -1.0, EMPTY
                put(lp+"cw", cw); put(lp+"tw", tw)
                for kk in MC_KEYS: put(lp+kk, info[kk])

out = {k: np.array(v, dtype=np.float64) for k, v in R.items()}
out.update({k: np.array(v, dtype=np.float64) for k, v in V.items()})
out["meta_infile"] = np.array(a.infile)
out["meta_events_read"] = np.array(n_ev)
out["meta_events_with_cand"] = np.array(n_ev_cand)
out["meta_cands_not_2_daughters"] = np.array(n_bad)
np.savez_compressed(a.outfile, **out)

n = len(R["evt"])
print(f"events read {n_ev}, with cand {n_ev_cand} ({100*n_ev_cand/max(n_ev,1):.1f}%), "
      f"candidates {n}, skipped {n_bad}")
if n:
    pdg = np.concatenate([out["d0_l0_pdg"], out["d1_l0_pdg"]])
    print("daughters with no link:", np.mean(pdg == 0))
    print("top link photon (22):", np.mean(pdg == 22))
    print("top link e+-  (11):", np.mean(np.abs(pdg) == 11))
    print("top link other:", np.mean((pdg != 0) & (pdg != 22) & (np.abs(pdg) != 11)))
    print("same top MC photon (both link, same idx):",
          np.mean((out["d0_l0_idx"] == out["d1_l0_idx"]) & (out["d0_l0_idx"] >= 0)))
