#!/usr/bin/env python3
"""
make_pi0_tables.py -- one ROOT pass -> one NPZ per file with two linked tables (+ a tiny third one).
RUN WITH NO ARGUMENTS: both files, all events. Test: python make_pi0_tables.py --max-events 1000

WHAT IT WRITES (prefix = table):
  pfo_*   REC side : one row per Pandora PFO with PDG 22, NO cuts.
                     Each row keeps its two best truth links (ranked by cluster weight).
  mc_*    MC side  : one row per MC pi0 (PDG 111) with exactly two daughters, both PDG 22. NO genStatus cut:
                     filter later on pi0_gen / ph0_gen / ph1_gen (generator pi0s have photon gen = 1).
                     For each of its two photons, the two best PFOs are stored.
  cand_*  finder   : one row per GammaGammaCandidatePi0s candidate (daughter PFO indices only).
  ev_*    per-event arrays for ALL events (also events with no pi0 / no candidate).
  meta_*  counters and the input file name.

CONVENTIONS (same as the old candidate cache unless stated):
  - cw / tw : decoded RecoMCTruthLink weight. raw=int(round(w)); cw=(raw//10000)/1000; tw=(raw%10000)/1000.
  - ASSUMPTION: cw = share of the PFO's cluster energy that comes from that MC particle.
  - Photon 0 of a pi0 = higher TRUE energy, photon 1 = lower. (ordering only)
  - MC -> PFO ranking ("best PFO of a photon") = cw * E_pfo  (energy that the PFO takes from this photon),
    NOT cw alone: a tiny fragment with cw=1.0 must not beat the main PFO with cw=0.8.
    The cw of every stored PFO is kept, so a cw-based reading is still possible for the stored slots.
  - Endpoint exactly (0,0,0) is stored as NaN (unknown), r_end = sqrt(x^2+y^2) in mm.
  - Angles in degrees (theta, phi), true_open_angle in rad, energies in GeV.
  - Missing link / missing PFO: cw = tw = -1, indices -1, pdg 0, floats NaN.
  - IDs are stored as integers. Four-momenta are float64 (mass = sqrt(E^2-p^2) cancels badly in float32).
  - Indices are LOCAL to one file and one event. Never match indices between the G4 and CC3 files.
  - No weight floor is applied here. Floors (0.5, 0.1) are chosen later, in Python.
"""
import argparse
import math
import os
import sys
import time
from collections import Counter

import numpy as np

I8, I32, I64, F32, F64 = np.int8, np.int32, np.int64, np.float32, np.float64
NAN = float("nan")


# ----------------------------------------------------------------------------
# 1. SCHEMAS  (name, dtype, default for "missing")
# ----------------------------------------------------------------------------
def link_target_cols(pre):
    return [
        (pre + "cw", F32, -1.0), (pre + "tw", F32, -1.0),
        (pre + "idx", I32, -1), (pre + "pdg", I32, 0), (pre + "gen", I32, -1),
        (pre + "E", F32, NAN), (pre + "theta", F32, NAN), (pre + "phi", F32, NAN),
        (pre + "fast", I8, -1), (pre + "r_end", F32, NAN), (pre + "z_end", F32, NAN),
        (pre + "n_dau", I32, -1), (pre + "n_par", I32, -1),
        (pre + "par0_pdg", I32, 0), (pre + "par0_idx", I32, -1), (pre + "par1_pdg", I32, 0),
        (pre + "pi0_par_idx", I32, -1), (pre + "any_par_e", I8, -1),
    ]


def pfo_schema():
    s = [("loop_evt", I64, -1), ("run", I64, -1), ("evt", I64, -1), ("pfo_idx", I32, -1),
         ("pdg", I32, 0), ("E", F64, NAN), ("px", F64, NAN), ("py", F64, NAN), ("pz", F64, NAN),
         ("n_links", I32, 0), ("sum_cw", F32, NAN), ("n_cand", I32, 0)]
    for j in (0, 1):
        s += link_target_cols("l%d_" % j)
    return s


def mc_schema():
    s = [("loop_evt", I64, -1), ("run", I64, -1), ("evt", I64, -1), ("pi0_idx", I32, -1),
         ("pi0_gen", I32, -1), ("pi0_par0_pdg", I32, 0),
         ("pi0_E", F64, NAN), ("pi0_px", F64, NAN), ("pi0_py", F64, NAN), ("pi0_pz", F64, NAN),
         ("pi0_theta", F32, NAN), ("pi0_phi", F32, NAN),
         ("true_pair_E", F64, NAN), ("true_open_angle", F64, NAN)]
    for j in (0, 1):
        p = "ph%d_" % j
        s += [(p + "idx", I32, -1), (p + "pdg", I32, 0), (p + "gen", I32, -1), (p + "E", F64, NAN),
              (p + "theta", F32, NAN), (p + "phi", F32, NAN), (p + "fast", I8, -1),
              (p + "any_par_e", I8, -1), (p + "n_dau", I32, -1),
              (p + "r_end", F32, NAN), (p + "z_end", F32, NAN),
              (p + "n_pfo_linked", I32, 0), (p + "sum_contrib", F32, NAN)]
        for k in (0, 1):
            q = p + "b%d_" % k
            s += [(q + "pfo_idx", I32, -1), (q + "pdg", I32, 0), (q + "E", F64, NAN),
                  (q + "px", F64, NAN), (q + "py", F64, NAN), (q + "pz", F64, NAN),
                  (q + "cw", F32, -1.0), (q + "tw", F32, -1.0), (q + "contrib", F32, NAN)]
    return s


def cand_schema():
    return [("loop_evt", I64, -1), ("run", I64, -1), ("evt", I64, -1), ("cand_idx", I32, -1),
            ("n_dau", I32, -1), ("d0_pfo_idx", I32, -1), ("d1_pfo_idx", I32, -1),
            ("c_E", F64, NAN), ("c_px", F64, NAN), ("c_py", F64, NAN), ("c_pz", F64, NAN)]


class Table:
    def __init__(self, schema):
        self.schema = schema
        self.cols = {n: [] for n, _, _ in schema}
        self.n = 0

    def add(self, row):
        extra = set(row) - set(self.cols)
        if extra:
            raise KeyError("unknown column(s): %s" % sorted(extra))
        for n, _, d in self.schema:
            self.cols[n].append(row.get(n, d))
        self.n += 1

    def arrays(self, prefix):
        return {prefix + n: np.asarray(self.cols[n], dtype=dt) for n, dt, _ in self.schema}


# ----------------------------------------------------------------------------
# 2. SMALL PURE FUNCTIONS (all tested on synthetic data)
# ----------------------------------------------------------------------------
def decode_weight(w):
    raw = int(round(w))
    return (raw // 10000) / 1000.0, (raw % 10000) / 1000.0


def theta_phi_deg(px, py, pz):
    p = math.sqrt(px * px + py * py + pz * pz)
    if p <= 0:
        return NAN, NAN
    return math.degrees(math.acos(max(-1.0, min(1.0, pz / p)))), math.degrees(math.atan2(py, px))


def open_angle(a, b):
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    cross = math.sqrt(cx * cx + cy * cy + cz * cz)
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return math.atan2(cross, dot)


def index_links(links, cnt):
    """links: iterable of (rec_idx, sim_idx, cw, tw). Drops exact (rec,sim) duplicates (counted)."""
    seen, by_rec, by_sim = set(), {}, {}
    for rec, sim, cw, tw in links:
        if (rec, sim) in seen:
            cnt["links_duplicate_dropped"] += 1
            continue
        seen.add((rec, sim))
        by_rec.setdefault(rec, []).append((sim, cw, tw))
        by_sim.setdefault(sim, []).append((rec, cw, tw))
        cnt["links_total"] += 1
    return by_rec, by_sim


def mc_props(idx, get_mc, get_pdg):
    """Properties of one MC particle. None if it cannot be read."""
    m = get_mc(idx)
    if m is None:
        return None
    theta, phi = theta_phi_deg(m["px"], m["py"], m["pz"])
    if m["x"] == 0.0 and m["y"] == 0.0 and m["z"] == 0.0:
        r_end = z_end = NAN
    else:
        r_end, z_end = math.hypot(m["x"], m["y"]), m["z"]
    par = list(m["parents"])
    pdgs = [get_pdg(i) for i in par]
    pi0_par = next((i for i, pd in zip(par, pdgs) if pd == 111), -1)
    return dict(idx=idx, pdg=m["pdg"], gen=m["gen"], E=m["E"], theta=theta, phi=phi,
                fast=m["fast"], r_end=r_end, z_end=z_end, n_dau=m["n_dau"], n_par=len(par),
                par0_pdg=pdgs[0] if len(pdgs) > 0 else 0, par0_idx=par[0] if len(par) > 0 else -1,
                par1_pdg=pdgs[1] if len(pdgs) > 1 else 0, pi0_par_idx=pi0_par,
                any_par_e=int(any(abs(pd) == 11 for pd in pdgs)))


def best_pfos_for_photon(mc_idx, by_sim, get_pfo, cnt, n=2):
    """Rank the PFOs linked to one MC photon by contribution = cw * E_pfo. Returns (top n, n_linked, sum_contrib)."""
    ent = []
    for rec, cw, tw in by_sim.get(mc_idx, []):
        p = get_pfo(rec)
        if p is None:
            cnt["links_bad_rec_index"] += 1
            continue
        ent.append((cw * p["E"], cw, tw, rec, p))
    ent.sort(key=lambda t: (-t[0], -t[1], t[3]))
    return ent[:n], len(ent), (sum(t[0] for t in ent) if ent else NAN)


# ----------------------------------------------------------------------------
# 3. ONE EVENT -> ROWS
# ----------------------------------------------------------------------------
def process_event(ev, tabs, cnt):
    """
    ev keys: loop_evt, run, evt,
             photon_pfo_idx : [int]                    PandoraPFOs with PDG 22
             links          : [(rec, sim, cw, tw)]     all RecoMCTruthLink entries (weights already decoded)
             pi0s           : [(mc_idx, [dau_idx])]    every MC particle with PDG 111
             cands          : [(cand_idx, [pfo_idx], (E,px,py,pz))]
             get_pfo(i) -> dict(pdg,E,px,py,pz) or None
             get_mc(i)  -> dict(pdg,gen,E,px,py,pz,fast,n_dau,parents,x,y,z) or None
             get_pdg(i) -> int
    Returns the per-event meta dict.
    """
    get_pfo, get_mc, get_pdg = ev["get_pfo"], ev["get_mc"], ev["get_pdg"]
    key = dict(loop_evt=ev["loop_evt"], run=ev["run"], evt=ev["evt"])
    by_rec, by_sim = index_links(ev["links"], cnt)

    n_cand_of = Counter()
    for ci, dau, fv in ev["cands"]:
        for d in dau:
            n_cand_of[d] += 1
        cnt["cand_total"] += 1
        if len(dau) != 2:
            cnt["cand_not_2_daughters"] += 1
        order = sorted(dau, key=lambda i: -(get_pfo(i)["E"] if get_pfo(i) else -1.0))
        row = dict(key, cand_idx=ci, n_dau=len(dau), c_E=fv[0], c_px=fv[1], c_py=fv[2], c_pz=fv[3])
        if len(order) > 0:
            row["d0_pfo_idx"] = order[0]
        if len(order) > 1:
            row["d1_pfo_idx"] = order[1]
        tabs["cand"].add(row)

    for pidx in ev["photon_pfo_idx"]:
        p = get_pfo(pidx)
        lk = sorted(by_rec.get(pidx, []), key=lambda t: (-t[1], -t[2], t[0]))
        row = dict(key, pfo_idx=pidx, pdg=p["pdg"], E=p["E"], px=p["px"], py=p["py"], pz=p["pz"],
                   n_links=len(lk), sum_cw=(sum(t[1] for t in lk) if lk else NAN),
                   n_cand=n_cand_of.get(pidx, 0))
        for j, (sim, cw, tw) in enumerate(lk[:2]):
            mp = mc_props(sim, get_mc, get_pdg)
            if mp is None:
                cnt["links_bad_sim_index"] += 1
                continue
            row["l%d_cw" % j], row["l%d_tw" % j] = cw, tw
            for k, v in mp.items():
                row["l%d_%s" % (j, k)] = v
        tabs["pfo"].add(row)
        cnt["pfo_rows"] += 1

    n_pi0_rows = 0
    for pi0_idx, dau in ev["pi0s"]:
        cnt["pi0_total"] += 1
        pm = get_mc(pi0_idx)
        if len(dau) != 2:
            cnt["pi0_skip_not_two_daughters"] += 1
            cnt["pi0_skip_not_two_daughters_pi0gen_%d" % pm["gen"]] += 1
            continue
        d = [mc_props(i, get_mc, get_pdg) for i in dau]
        if any(x is None for x in d):
            cnt["pi0_skip_unreadable_daughter"] += 1
            continue
        if not (d[0]["pdg"] == 22 and d[1]["pdg"] == 22):
            cnt["pi0_skip_not_two_photons"] += 1
            cnt["pi0_skip_not_two_photons_pi0gen_%d" % pm["gen"]] += 1
            continue
        d.sort(key=lambda x: (-x["E"], x["idx"]))
        m0, m1 = get_mc(d[0]["idx"]), get_mc(d[1]["idx"])
        theta, phi = theta_phi_deg(pm["px"], pm["py"], pm["pz"])
        ppdg = get_pdg(pm["parents"][0]) if pm["parents"] else 0
        row = dict(key, pi0_idx=pi0_idx, pi0_gen=pm["gen"], pi0_par0_pdg=ppdg, pi0_E=pm["E"],
                   pi0_px=pm["px"], pi0_py=pm["py"], pi0_pz=pm["pz"], pi0_theta=theta, pi0_phi=phi,
                   true_pair_E=d[0]["E"] + d[1]["E"],
                   true_open_angle=open_angle((m0["px"], m0["py"], m0["pz"]), (m1["px"], m1["py"], m1["pz"])))
        for j in (0, 1):
            pre = "ph%d_" % j
            for name in ("idx", "pdg", "gen", "E", "theta", "phi", "fast", "any_par_e", "n_dau", "r_end", "z_end"):
                row[pre + name] = d[j][name]
            top, n_linked, sum_c = best_pfos_for_photon(d[j]["idx"], by_sim, get_pfo, cnt)
            row[pre + "n_pfo_linked"], row[pre + "sum_contrib"] = n_linked, sum_c
            for k, (contrib, cw, tw, rec, pf) in enumerate(top):
                q = pre + "b%d_" % k
                row.update({q + "pfo_idx": rec, q + "pdg": pf["pdg"], q + "E": pf["E"], q + "px": pf["px"],
                            q + "py": pf["py"], q + "pz": pf["pz"], q + "cw": cw, q + "tw": tw,
                            q + "contrib": contrib})
        tabs["mc"].add(row)
        n_pi0_rows += 1
        cnt["pi0_rows"] += 1
        cnt["pi0_rows_photon_gens_%d_%d" % tuple(sorted((d[0]["gen"], d[1]["gen"])))] += 1

    return dict(loop=ev["loop_evt"], run=ev["run"], evt=ev["evt"], n_pi0=len(ev["pi0s"]),
                n_pi0_rows=n_pi0_rows, n_photon_pfo=len(ev["photon_pfo_idx"]), n_cands=len(ev["cands"]))


def run_events(events, report_every=0):
    tabs = {"pfo": Table(pfo_schema()), "mc": Table(mc_schema()), "cand": Table(cand_schema())}
    cnt, meta, t0 = Counter(), [], time.time()
    for ev in events:
        meta.append(process_event(ev, tabs, cnt))
        cnt["events_read"] += 1
        if report_every and cnt["events_read"] % report_every == 0:
            print("  %d events, %.0f s, pi0 rows %d, pfo rows %d" %
                  (cnt["events_read"], time.time() - t0, tabs["mc"].n, tabs["pfo"].n), flush=True)
    return tabs, meta, cnt


def save_npz(path, tabs, meta, cnt, infile, tag):
    out = {}
    for name, pre in (("pfo", "pfo_"), ("mc", "mc_"), ("cand", "cand_")):
        out.update(tabs[name].arrays(pre))
    for k in ("loop", "run", "evt", "n_pi0", "n_pi0_rows", "n_photon_pfo", "n_cands"):
        out["ev_" + k] = np.asarray([m[k] for m in meta], dtype=I64)
    for k, v in cnt.items():
        out["meta_" + k] = np.asarray(v)
    out["meta_infile"], out["meta_tag"] = np.asarray(infile), np.asarray(tag)
    np.savez_compressed(path, **out)


def print_summary(tabs, cnt):
    print("\n=== SUMMARY ===")
    for k in sorted(cnt):
        print("  %-44s %d" % (k, cnt[k]))
    mc = tabs["mc"]
    if mc.n:
        gen = np.asarray(mc.cols["ph0_gen"]) == 1
        gen &= np.asarray(mc.cols["ph1_gen"]) == 1
        print("  mc rows with both photons genStatus 1 (generator pi0s): %d of %d" % (gen.sum(), mc.n))
        for j in (0, 1):
            sc = np.asarray(mc.cols["ph%d_sum_contrib" % j], dtype=float)
            e = np.asarray(mc.cols["ph%d_E" % j], dtype=float)
            ok = np.isfinite(sc)
            print("  photon %d: with >=1 linked PFO %.1f%%; median sum_contrib/E_true %.3f "
                  "(about 1.0-1.03 expected: the energy response medians are 1.007-1.03)" %
                  (j, 100.0 * ok.mean(), float(np.median(sc[ok] / e[ok])) if ok.any() else NAN))
    print("  Counters that do not appear are zero. links_bad_rec_index / links_bad_sim_index must be 0.")


# ----------------------------------------------------------------------------
# 4. READING LAYER (podio / EDM4hep). Worked on the real 1k test (G4 and CC3).
# ----------------------------------------------------------------------------
def _link_ends(link):
    if hasattr(link, "getRec"):
        return link.getRec(), link.getSim()
    return link.getFrom(), link.getTo()


def _pfo_pdg(p):
    return p.getPDG() if hasattr(p, "getPDG") else p.getType()


def _open_reader(path):
    from podio import root_io
    return root_io.Reader(path) if hasattr(root_io, "Reader") else root_io.RootReader(path)


def read_events(path, names, max_events, cnt_io):
    reader = _open_reader(path)
    for loop_evt, frame in enumerate(reader.get("events")):
        if max_events and loop_evt >= max_events:
            break
        hdr = frame.get(names["hdr"])[0]
        pfos, mcs = frame.get(names["pfo"]), frame.get(names["mc"])
        links, cands = frame.get(names["link"]), frame.get(names["cand"])
        n_pfo, n_mc = len(pfos), len(mcs)
        pfo_cid = pfos[0].getObjectID().collectionID if n_pfo else None
        mc_cache, pfo_cache = {}, {}

        def get_pfo(i, pfos=pfos, cache=pfo_cache, n=n_pfo):
            if i in cache:
                return cache[i]
            if not (0 <= i < n):
                cache[i] = None
                return None
            p = pfos[i]
            m = p.getMomentum()
            cache[i] = dict(pdg=_pfo_pdg(p), E=p.getEnergy(), px=m.x, py=m.y, pz=m.z)
            return cache[i]

        def get_mc(i, mcs=mcs, cache=mc_cache, n=n_mc):
            if i in cache:
                return cache[i]
            if not (0 <= i < n):
                cache[i] = None
                return None
            p = mcs[i]
            m, e = p.getMomentum(), p.getEndpoint()
            en = p.getEnergy() if hasattr(p, "getEnergy") else math.sqrt(m.x**2 + m.y**2 + m.z**2 + p.getMass()**2)
            try:
                fast = int(p.isHandledByFastSim())
            except Exception:
                fast = -1
            cache[i] = dict(pdg=p.getPDG(), gen=p.getGeneratorStatus(), E=en, px=m.x, py=m.y, pz=m.z, fast=fast,
                            n_dau=len(p.getDaughters()), parents=[q.getObjectID().index for q in p.getParents()],
                            x=e.x, y=e.y, z=e.z)
            return cache[i]

        def get_pdg(i, mcs=mcs, n=n_mc):
            return mcs[i].getPDG() if 0 <= i < n else 0

        photon_idx = []
        for pos, p in enumerate(pfos):
            if p.getObjectID().index != pos:
                cnt_io["pfo_index_not_position"] += 1
            if _pfo_pdg(p) == 22:
                photon_idx.append(pos)
        lk = []
        for l in links:
            r, s = _link_ends(l)
            cw, tw = decode_weight(l.getWeight())
            lk.append((r.getObjectID().index, s.getObjectID().index, cw, tw))
        pi0s = [(pos, [q.getObjectID().index for q in m.getDaughters()])
                for pos, m in enumerate(mcs) if m.getPDG() == 111]
        cl = []
        for ci, c in enumerate(cands):
            dts = list(c.getParticles())
            for q in dts:
                if pfo_cid is not None and q.getObjectID().collectionID != pfo_cid:
                    cnt_io["cand_daughter_not_in_pfo_collection"] += 1
            m = c.getMomentum()
            cl.append((ci, [q.getObjectID().index for q in dts], (c.getEnergy(), m.x, m.y, m.z)))
        yield dict(loop_evt=loop_evt, run=hdr.getRunNumber(), evt=hdr.getEventNumber(),
                   photon_pfo_idx=photon_idx, links=lk, pi0s=pi0s, cands=cl,
                   get_pfo=get_pfo, get_mc=get_mc, get_pdg=get_pdg)


# ----------------------------------------------------------------------------
# 5. HARD-CODED FILES AND OUTPUT DIRECTORY  (edit here if paths change)
# ----------------------------------------------------------------------------
REC_DIR = "/data/dust/user/alimuham/thesis/reco"
OUT_DIR = REC_DIR
JOBS = {
    "g4": (REC_DIR + "/tau-pi0-geant4-100kevents-reco-test_REC.edm4hep.root", "G4"),
    "cc3": (REC_DIR + "/tau-pi0-caloclouds-100kevents-reco-test_REC.edm4hep.root", "CC3"),
}
NAMES = dict(pfo="PandoraPFOs", mc="MCParticles", link="RecoMCTruthLink",
             cand="GammaGammaCandidatePi0s", hdr="EventHeader")


def main():
    ap = argparse.ArgumentParser(description="Run with no arguments: both files, all events.")
    ap.add_argument("--which", choices=["g4", "cc3", "both"], default="both")
    ap.add_argument("--max-events", type=int, default=0, help="0 = all events (test: 1000)")
    ap.add_argument("--outdir", default=OUT_DIR)
    ap.add_argument("--report-every", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    suffix = "full" if not a.max_events else "first%d" % a.max_events
    for key in (["g4", "cc3"] if a.which == "both" else [a.which]):
        infile, tag = JOBS[key]
        out = os.path.join(a.outdir, "pi0-tables-%s-%s.npz" % (key, suffix))
        print("\n##### %s : %s\n      -> %s" % (tag, infile, out), flush=True)
        cnt_io = Counter()
        tabs, meta, cnt = run_events(read_events(infile, NAMES, a.max_events, cnt_io), a.report_every)
        cnt.update(cnt_io)
        print_summary(tabs, cnt)
        save_npz(out, tabs, meta, cnt, infile, tag)
        print("saved", out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
