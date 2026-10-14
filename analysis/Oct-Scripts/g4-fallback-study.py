#!/usr/bin/env python3
"""
fallback_study_v1.py   (v1.1, after stress test)
================================================
Purpose: understand what the "G4-fallback" photons are (Geant4 photons inside the
CC3 run) and how much they contribute. ONLY for that study; separate from
single_photon_observables_v5.py on purpose.

One row per TRUTH photon (Layer A + Layer B), lost or not. Three groups:
    G4  : G4 file,  A+B (no flag used to select)
    CC3 : CC3 file, A+B, isHandledByFastSim() == True   (True-CC3)
    FB  : CC3 file, A+B, isHandledByFastSim() == False  (G4-fallback)
Layer A / Layer B are IDENTICAL to v5.1 (theta 40-140 deg, phi dead window
(C-0.01-0.5, C+8+0.5) deg, status 1, E>=10 GeV, no e+- parent among ANY parent).

Links: RecoMCTruthLink, decoded weights cw = (raw // 10000)/1000 (CLUSTER weight, the
gate used everywhere) and tw = (raw % 10000)/1000 (TRACK weight, stored, not a gate).
Floors 0.1 and 0.5 are evaluated in the SAME pass (suffix f10 / f50).

Link category per truth photon (priority order, at each floor f, on cluster weight):
    1 owned     : a photon PFO (PDG 22) has THIS photon as its best link (cw >= f)
    2 shared    : a photon PFO links to it with cw >= f, but that PFO's best link is another MC
    3 nonphoton : only non-photon PFOs (electron, neutral hadron, ...) link with cw >= f
    4 desc      : no link to the photon itself, but a PFO links (cw >= f) to a DESCENDANT
    5 anc       : no link to photon/descendants, but to an ANCESTOR (parent chain)
    6 none      : no PFO link >= f to the photon, its descendants or its ancestors
Category 1 at f=0.5 is exactly the v5.1 "matched" set (same best-link rule), so
its share per group MUST equal v5.1's printed matched fraction (cross-check).
Category 6 is checked for hidden TRACK-only links (printed table), because a charged
PFO (e.g. from a converted photon) may carry its weight in tw, not cw.
Floor-free values (max cw / max tw per class, PDG of the best link) are saved so any
other floor can be explored offline.

Definitions (nothing implicit)
    completeness_own_f = sum over links to THIS photon with cw >= f of (cw * E_pfo) / E_mc
    completeness_fam_f = completeness_own_f + same sum over links to its DESCENDANTS
    n_split_any_f / n_split_ph_f = links with cw >= f to the photon, any PFO type / photon PFOs
    nn1/nn5_angle = smallest 3D angle [rad] to any OTHER status-1 non-neutrino truth particle
                    with E >= 1 / 5 GeV in the same event (not Delta R)
    phi_crack_dist = signed distance [deg] to the nearest crack centre,
                     raw = ((phi - C + 180) % 360) - 180   (dead zone is 0..8, removed)
    sib_* = another selected photon with the same first parent, and its flag (CC3 file only)
    r_end = sqrt(x^2 + y^2) of mc.getEndpoint() [mm]; endpoint (0,0,0) is treated as not stored

Outputs (folder ~/thesis-ml-sim/plots/<group>/fallback_study_v1_<dataset>/)
    fallback_photons_<dataset>.npz  one row per truth photon, all columns
    summary.txt                      every printed table
    figA_population_check.png        E, cos(theta), phi of the 3 groups (unit area) + fallback share
    figB_lost_fraction.png           not-owned and none fractions vs E_mc, both floors, binomial errors
    figC_link_categories.png         category shares per group, both floors
    figD_completeness.png            completeness distributions (own / incl. descendants)

Run
    source ~/source.sh
    python3 fallback_study_v1.py --dataset 1k
    nohup python3 fallback_study_v1.py --dataset 100k > fallback_100k.log 2>&1 &
Options: --group NAME, --ecal-r MM (only for the printed endpoint classes; raw r_end saved)

STATUS: tested on mock objects only (logic, categories, tables, figures). NOT run on the
real files. Check on 1k first. Unverified until then:
    - REC-level MCParticles keep daughters / parents / endpoints (print-out reports it)
    - default --ecal-r 1800 mm is from memory; the True-CC3 r_end percentiles printed in the
      summary show where the fast-sim endpoint really sits. The ECAL barrel is polygonal,
      so the front radius varies with phi: treat the endpoint classes as coarse.

OPEN CHECKLIST (kept open on purpose, not part of this script)
    [ ] Geometry: window kept at theta 40-140 + 0.5 deg phi margin ("safest for now").
        Old numbers (84.4% precision, 58.7% zero-link at 10k, completeness 0.396) used
        37.5-142.5 and no margin => SUPERSEDED, not comparable. Revisit 37.5-142.5 later;
        compare the fallback share printed here with the old 15.6%.
    [ ] Weight floor: studied at 0.1 AND 0.5 here. v5.1 still uses 0.5 for n_split; make a
        separate split floor (0.1) there when the single-photon plots are redone.
    [ ] Radial profile: axis is still the reco hit centroid; MC-direction axis untested.
        Normalisation pools ring energy over shower energy within an E bin (high-energy
        showers weigh more). One sentence on the plot when revisited.
    [ ] v5.1 n_split denominator = photons with >=1 link of ANY type (not all truth photons).
    [ ] v5.1 fake rate drops PFOs below 10 GeV; a converted photon counts as fake.
    [ ] Ask Thomas which conditions make isHandledByFastSim() False for an eligible photon.
"""
import os
import sys
import math
import argparse
from collections import defaultdict, Counter
import numpy as np

PHOTON_PDG, ELECTRON_PDG = 22, 11
NEUTRINOS = {12, 14, 16}
E_MC_MIN = 10.0
THETA_LO, THETA_HI = 40.0, 140.0
CRACKS = [-157.5 + 45.0 * k for k in range(8)]
_A, _B = -0.01 - 0.5, 8.0 + 0.5
FLOORS = (("f10", 0.1), ("f50", 0.5))
NN_THR = (("nn1", 1.0), ("nn5", 5.0))
DESC_CAP, ANC_CAP = 300, 30
GROUP_LABEL = {0: "Geant4 (all selected)", 1: "True-CC3", 2: "G4-fallback"}
GROUP_COLOR = {0: "#2166ac", 1: "#d6604d", 2: "#7f7f7f"}
E_BINS = np.array([10, 20, 30, 40, 60, 80, 100, 125], dtype=float)
THETA_FOLD_BINS = np.array([40, 41, 42, 45, 50, 60, 70, 80, 90.01])
PHI_DIST_BINS = np.array([-22.5, -10, -5, -2, -0.5, 8.5, 10, 12, 15, 20, 22.5])
ENDPOINT_MARGIN_MM = 20.0
MIN_PLOT_N = 5
NOTE = "status 1, E ≥ 10 GeV, no e± parent | barrel θ 40–140°, φ dead zones removed"

LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def oid(obj):
    o = obj.getObjectID()
    try:
        return (int(o.collectionID), int(o.index))
    except AttributeError:
        return (0, int(o.index))


def decode_weight(raw_weight):
    raw_int = int(round(raw_weight))
    return (raw_int // 10000) / 1000.0, (raw_int % 10000) / 1000.0


def in_dead_window(phi_deg):
    for c in CRACKS:
        if ((phi_deg - c - _A) % 360.0) <= (_B - _A):
            return True
    return False


def crack_signed_distance(phi_deg):
    raws = [((phi_deg - c + 180.0) % 360.0) - 180.0 for c in CRACKS]
    return min(raws, key=abs)


def theta_phi(mc):
    m = mc.getMomentum()
    p = math.sqrt(m.x ** 2 + m.y ** 2 + m.z ** 2)
    if p == 0:
        return float("nan"), float("nan")
    return (math.degrees(math.acos(max(-1.0, min(1.0, m.z / p)))),
            math.degrees(math.atan2(m.y, m.x)))


def layer_a(mc):
    if mc.getGeneratorStatus() != 1 or mc.getPDG() != PHOTON_PDG:
        return False
    if mc.getEnergy() < E_MC_MIN:
        return False
    return not any(abs(q.getPDG()) == ELECTRON_PDG for q in mc.getParents())


def layer_b(theta, phi):
    if not (THETA_LO <= theta <= THETA_HI):
        return False
    return not in_dead_window(phi)


def walk(mc, method, cap):
    seen, stack = {}, [mc]
    while stack and len(seen) < cap:
        cur = stack.pop()
        for q in getattr(cur, method)():
            k = oid(q)
            if k not in seen:
                seen[k] = q
                stack.append(q)
    return seen


def first_parent(mc):
    ps = list(mc.getParents())
    return ps[0] if ps else None


def nearest_other(i_self, ps, es, pdgs, pars, thr):
    mask = es >= thr
    mask[i_self] = False
    p0 = ps[i_self]
    n0 = np.linalg.norm(p0)
    sub = np.where(mask)[0]
    if n0 == 0 or len(sub) == 0:
        return float("nan"), 0, 0
    norms = np.linalg.norm(ps[sub], axis=1)
    ok = norms > 0
    if not ok.any():
        return float("nan"), 0, 0
    sub, norms = sub[ok], norms[ok]
    ang = np.arccos(np.clip((ps[sub] @ p0) / (norms * n0), -1.0, 1.0))
    j = int(np.argmin(ang))
    k = sub[j]
    sib = int(pars[i_self] is not None and pars[k] == pars[i_self])
    return float(ang[j]), int(pdgs[k]), sib


def link_summary(mid, own, fam, best_f, f, e_mc):
    owned = [l for l in own if l[1] == PHOTON_PDG and l[3] >= f
             and best_f.get(l[0], (None, 0))[0] == mid]
    shared = [l for l in own if l[1] == PHOTON_PDG and l[3] >= f
              and best_f.get(l[0], (None, 0))[0] != mid]
    nonph = [l for l in own if l[1] != PHOTON_PDG and l[3] >= f]
    desc_hit = [l for l in fam["desc"] if l[3] >= f]
    anc_hit = [l for l in fam["anc"] if l[3] >= f]
    if owned:
        cat = 1
    elif shared:
        cat = 2
    elif nonph:
        cat = 3
    elif desc_hit:
        cat = 4
    elif anc_hit:
        cat = 5
    else:
        cat = 6
    own_f = [l for l in own if l[3] >= f]
    comp_own = sum(l[3] * l[2] for l in own_f) / e_mc
    comp_fam = comp_own + sum(l[3] * l[2] for l in desc_hit) / e_mc
    if owned:
        top = max(owned, key=lambda l: l[2])
        owned_e, owned_cw = top[2] / e_mc, top[3]
    else:
        owned_e, owned_cw = float("nan"), float("nan")
    return dict(cat=cat, owned_e=owned_e, owned_cw=owned_cw, comp_own=comp_own,
                comp_fam=comp_fam, nsplit_any=len(own_f),
                nsplit_ph=sum(1 for l in own_f if l[1] == PHOTON_PDG))


def max_cw(links, pred=lambda l: True):
    v = [l for l in links if pred(l)]
    if not v:
        return 0.0, 0
    top = max(v, key=lambda l: l[3])
    return top[3], top[1]


def process_event(event, is_cc3, n_ev, rows, stats):
    mc_all = event.get("MCParticles")
    for _m in mc_all:
        stats["mc_coll"] = oid(_m)[0]
        break
    links_by_mc = defaultdict(list)
    best = {t: {} for t, _ in FLOORS}
    for link in event.get("RecoMCTruthLink"):
        cw, tw = decode_weight(link.getWeight())
        pfo = link.getFrom()
        pid = oid(pfo)
        mk = oid(link.getTo())
        links_by_mc[mk].append((pid, pfo.getPDG(), pfo.getEnergy(), cw, tw))
        stats["links"] += 1
        stats["links_resolved"] += int(mk[0] == stats.get("mc_coll", mk[0]))
        for t, f in FLOORS:
            if cw >= f:
                cur = best[t].get(pid)
                if cur is None or cw > cur[1]:
                    best[t][pid] = (mk, cw)

    pdgs, es, ps, pars, sel = [], [], [], [], []
    for mc in mc_all:
        if mc.getGeneratorStatus() != 1 or abs(mc.getPDG()) in NEUTRINOS:
            continue
        mom = mc.getMomentum()
        fp = first_parent(mc)
        pdgs.append(mc.getPDG())
        es.append(mc.getEnergy())
        ps.append((mom.x, mom.y, mom.z))
        pars.append(oid(fp) if fp is not None else None)
        if layer_a(mc):
            th, ph = theta_phi(mc)
            if math.isfinite(th) and layer_b(th, ph):
                sel.append((mc, th, ph, len(pdgs) - 1))
    if not sel:
        return
    es_a, ps_a = np.array(es, dtype=float), np.array(ps, dtype=float)

    flags = []
    for mc, _, _, _ in sel:
        try:
            flags.append(int(bool(mc.isHandledByFastSim())))
        except AttributeError:
            sys.exit("isHandledByFastSim() not found. Run 'source ~/source.sh' (Key4hep nightlies).")
    parent_ids = []
    for mc, _, _, _ in sel:
        fp = first_parent(mc)
        parent_ids.append(oid(fp) if fp is not None else None)

    for n, (mc, th, ph, idx) in enumerate(sel):
        mid = oid(mc)
        e_mc = mc.getEnergy()
        flag = flags[n]
        group = (1 if flag else 2) if is_cc3 else 0
        own = links_by_mc.get(mid, [])
        desc = walk(mc, "getDaughters", DESC_CAP)
        anc = walk(mc, "getParents", ANC_CAP)
        fam = dict(desc=[l for k in desc for l in links_by_mc.get(k, [])],
                   anc=[l for k in anc for l in links_by_mc.get(k, [])])
        fp = first_parent(mc)
        gp = first_parent(fp) if fp is not None else None
        daus = list(mc.getDaughters())
        try:
            ep, vx = mc.getEndpoint(), mc.getVertex()
            r_end, z_end = math.hypot(ep.x, ep.y), ep.z
            r_vtx = math.hypot(vx.x, vx.y)
            if ep.x == 0 and ep.y == 0 and ep.z == 0:
                r_end = z_end = float("nan")
        except AttributeError:
            r_end = z_end = r_vtx = float("nan")
        sib = [m for m in range(len(sel)) if m != n and parent_ids[n] is not None
               and parent_ids[m] == parent_ids[n]]
        row = dict(group=group, event=n_ev, mc_index=mid[1], E_mc=e_mc, theta=th, phi=ph,
                   phi_crack_dist=crack_signed_distance(ph), flag=flag,
                   parent_pdg=abs(fp.getPDG()) if fp is not None else 0,
                   grandparent_pdg=abs(gp.getPDG()) if gp is not None else 0,
                   n_parents=len(list(mc.getParents())), n_dau=len(daus),
                   n_dau_e=sum(1 for q in daus if abs(q.getPDG()) == ELECTRON_PDG),
                   n_desc=len(desc), r_end=r_end, z_end=z_end, r_vtx=r_vtx,
                   sib_n=len(sib), sib_flag=flags[sib[0]] if sib else -1,
                   n_sel_event=len(sel))
        for tag, thr in NN_THR:
            a, p, s = nearest_other(idx, ps_a, es_a, pdgs, pars, thr)
            row[tag + "_angle"], row[tag + "_pdg"], row[tag + "_sib"] = a, p, s
        row["maxcw_phpfo"], _ = max_cw(own, lambda l: l[1] == PHOTON_PDG)
        row["maxcw_nonph"], _ = max_cw(own, lambda l: l[1] != PHOTON_PDG)
        row["maxcw_desc"], row["pdg_bestdesc"] = max_cw(fam["desc"])
        row["maxcw_anc"], _ = max_cw(fam["anc"])
        _, row["pdg_bestlink"] = max_cw(own)
        row["maxtw_own"] = max([l[4] for l in own], default=0.0)
        row["maxtw_desc"] = max([l[4] for l in fam["desc"]], default=0.0)
        row["maxtw_anc"] = max([l[4] for l in fam["anc"]], default=0.0)
        for t, f in FLOORS:
            for key, val in link_summary(mid, own, fam, best[t], f, e_mc).items():
                row[f"{key}_{t}"] = val
        rows.append(row)


def run(filepath, label, is_cc3, max_events):
    from podio import root_io
    log(f"\n{'=' * 60}\n  {label}: {filepath}\n{'=' * 60}")
    rows, n_ev = [], 0
    stats = dict(links=0, links_resolved=0)
    for event in root_io.Reader(filepath).get("events"):
        if n_ev >= max_events:
            break
        n_ev += 1
        process_event(event, is_cc3, n_ev, rows, stats)
        if n_ev % 10000 == 0:
            log(f"  processed {n_ev} events ({len(rows)} photon rows)")
    log(f"  Events processed: {n_ev}; truth photon rows: {len(rows)}")
    log(f"  Link targets inside the MCParticles collection: {stats['links_resolved']}/{stats['links']} "
        f"(must be ~100%, else links point to another collection and every photon looks lost)")
    if not is_cc3:
        log(f"  Photons flagged fast-sim in the G4 file (expected 0): "
            f"{sum(1 for r in rows if r['flag'])}")
    return rows


def to_arrays(rows):
    return {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}


def pct(n, d):
    return f"{100.0 * n / d:5.1f}%" if d else "   -  "


def share_table(d, name, x, edges):
    cc = np.isin(d["group"], (1, 2))
    log(f"\n  Fallback share of CC3-file photons vs {name}  [FB / (True-CC3 + FB)]")
    for i in range(len(edges) - 1):
        m = cc & (x >= edges[i]) & (x < edges[i + 1])
        n, nfb = int(m.sum()), int((m & (d["group"] == 2)).sum())
        log(f"    [{edges[i]:7.1f},{edges[i + 1]:7.1f})  N={n:7d}  FB={pct(nfb, n)}")


def summarize(d):
    n = {g: int((d["group"] == g).sum()) for g in (0, 1, 2)}
    log("\n=== Truth photons (Layer A+B) per group (compare with v5.1 'Selected truth photons') ===")
    for g in (0, 1, 2):
        log(f"  {GROUP_LABEL[g]:24s} {n[g]:9d}")
    log(f"  Fallback share of CC3-file photons: {pct(n[2], n[1] + n[2])}  "
        f"(old 15.6% used a wider window; not directly comparable)")

    log("\n=== Population check: where does fallback sit? ===")
    share_table(d, "E_mc [GeV]", d["E_mc"], E_BINS)
    share_table(d, "folded theta [deg] (90 = barrel centre)",
                np.minimum(d["theta"], 180.0 - d["theta"]), THETA_FOLD_BINS)
    share_table(d, "signed phi-distance to nearest crack [deg] (dead zone removed)",
                d["phi_crack_dist"], PHI_DIST_BINS)

    names = ["1 owned", "2 shared", "3 nonphoton", "4 via descendants", "5 via ancestors", "6 none"]
    for t, f in FLOORS:
        log(f"\n=== Link category per group, floor {f} (share of ALL truth photons) ===")
        log("  (category 1 at floor 0.5 must equal v5.1's matched fraction per group)")
        log("  " + "category".ljust(20) + "".join(GROUP_LABEL[g][:16].ljust(18) for g in (0, 1, 2)))
        for c in range(1, 7):
            line = "  " + names[c - 1].ljust(20)
            for g in (0, 1, 2):
                m = d["group"] == g
                k = int((m & (d["cat_" + t] == c)).sum())
                line += f"{pct(k, int(m.sum()))} (N={k})".ljust(18)
            log(line)
        log(f"  completeness (sum cw*E_pfo / E_mc), floor {f}: mean / median")
        for g in (0, 1, 2):
            m = d["group"] == g
            if m.sum():
                co, cf = d["comp_own_" + t][m], d["comp_fam_" + t][m]
                log(f"    {GROUP_LABEL[g]:24s} own {co.mean():.3f}/{np.median(co):.3f}   "
                    f"incl. descendants {cf.mean():.3f}/{np.median(cf):.3f}")
        log(f"  n_split (any PFO type), fraction >=2 among photons with >=1 link, floor {f}")
        for g in (0, 1, 2):
            m = (d["group"] == g) & (d["nsplit_any_" + t] >= 1)
            if m.sum():
                log(f"    {GROUP_LABEL[g]:24s} {pct(int((d['nsplit_any_' + t][m] >= 2).sum()), int(m.sum()))} (N={int(m.sum())})")

    log("\n=== Category 6 (none): hidden track-only links? (cluster weight is the gate) ===")
    for t, f in FLOORS:
        for g in (0, 1, 2):
            m = (d["group"] == g) & (d["cat_" + t] == 6)
            if m.sum():
                tr = np.maximum(np.maximum(d["maxtw_own"][m], d["maxtw_desc"][m]), d["maxtw_anc"][m])
                log(f"  floor {f}  {GROUP_LABEL[g]:24s} cat 6: N={int(m.sum()):7d}  "
                    f"with a track-weight link >= 0.1: {pct(int((tr >= 0.1).sum()), int(m.sum()))}")

    log("\n=== Origin: parent PDG (|PDG|) per group, top 6 ===")
    for g in (0, 1, 2):
        c = Counter(d["parent_pdg"][d["group"] == g].astype(int).tolist())
        tot = sum(c.values())
        log(f"  {GROUP_LABEL[g]:24s} " + ", ".join(f"{k}: {pct(v, tot).strip()}" for k, v in c.most_common(6)))

    log("\n=== Sibling photon (same first parent, also selected; CC3-file groups only) ===")
    for g in (1, 2):
        m = d["group"] == g
        if m.sum():
            s = d["sib_flag"][m]
            log(f"  {GROUP_LABEL[g]:24s} no sibling {pct(int((s == -1).sum()), int(m.sum()))}  "
                f"sibling True-CC3 {pct(int((s == 1).sum()), int(m.sum()))}  "
                f"sibling not-fast-sim {pct(int((s == 0).sum()), int(m.sum()))}")

    log("\n=== MC-tree availability and endpoint ===")
    ecal_r = d["_ecal_r"]
    for g in (0, 1, 2):
        m = d["group"] == g
        if m.sum() == 0:
            continue
        has_ep = np.isfinite(d["r_end"][m])
        med = np.nanmedian(d["r_end"][m]) if has_ep.any() else float("nan")
        log(f"  {GROUP_LABEL[g]:24s} endpoint stored {pct(int(has_ep.sum()), int(m.sum()))}  "
            f">=1 daughter {pct(int((d['n_dau'][m] >= 1).sum()), int(m.sum()))}  "
            f">=1 e+- daughter {pct(int((d['n_dau_e'][m] >= 1).sum()), int(m.sum()))}  "
            f"median r_end {med:.0f} mm")
    mt = d["group"] == 1
    if mt.any() and np.isfinite(d["r_end"][mt]).any():
        q = np.nanpercentile(d["r_end"][mt], [5, 25, 50, 75, 95])
        log("  True-CC3 r_end percentiles 5/25/50/75/95 [mm]: " + " / ".join(f"{v:.0f}" for v in q)
            + "   <- read the ECAL front from here (fast sim ends the track at the surface)")
    log(f"  endpoint classes use ECAL front radius {ecal_r:.0f} mm +- {ENDPOINT_MARGIN_MM:.0f} mm "
        f"(unverified default; raw r_end is saved; ECAL is polygonal so classes are coarse)")
    r = d["r_end"]
    cls = np.where(~np.isfinite(r), 0, np.where(r < ecal_r - ENDPOINT_MARGIN_MM, 1,
                   np.where(r <= ecal_r + ENDPOINT_MARGIN_MM, 2, 3)))
    cname = {1: "before ECAL", 2: "at ECAL front", 3: "inside/after ECAL"}
    for t, f in FLOORS:
        log(f"\n  Reconstruction outcome by endpoint class, floor {f}: share owned (cat 1) / none (cat 6)")
        for g in (0, 1, 2):
            for c in (1, 2, 3):
                m = (d["group"] == g) & (cls == c)
                if m.sum():
                    log(f"    {GROUP_LABEL[g]:24s} {cname[c]:18s} N={int(m.sum()):7d}  "
                        f"owned {pct(int((d['cat_' + t][m] == 1).sum()), int(m.sum()))}  "
                        f"none {pct(int((d['cat_' + t][m] == 6).sum()), int(m.sum()))}")

    log("\n=== Nearest other status-1 particle (3D angle) ===")
    for tag, thr in NN_THR:
        for g in (0, 1, 2):
            m = (d["group"] == g) & np.isfinite(d[tag + "_angle"])
            if m.sum():
                a = d[tag + "_angle"][m]
                log(f"  E>={thr:.0f} GeV  {GROUP_LABEL[g]:24s} median {np.median(a):.3f} rad  "
                    f"< 0.05 rad: {pct(int((a < 0.05).sum()), int(m.sum()))}  "
                    f"nearest is sibling: {pct(int((d[tag + '_sib'][m] == 1).sum()), int(m.sum()))}")


def _rate(x, num, den, edges):
    c, p, e, nn = [], [], [], []
    for i in range(len(edges) - 1):
        m = den & (x >= edges[i]) & (x < edges[i + 1])
        n = int(m.sum())
        if n >= MIN_PLOT_N:
            q = float((m & num).sum()) / n
            c.append(0.5 * (edges[i] + edges[i + 1]))
            p.append(q)
            e.append(math.sqrt(q * (1 - q) / n))
            nn.append(n)
    return np.array(c), np.array(p), np.array(e), np.array(nn)


def make_figures(d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    grp = d["group"]
    cos_t = np.cos(np.radians(d["theta"]))

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for g in (0, 1, 2):
        m = grp == g
        if m.sum() < MIN_PLOT_N:
            continue
        kw = dict(histtype="step", density=True, color=GROUP_COLOR[g], lw=2, label=GROUP_LABEL[g])
        ax[0, 0].hist(d["E_mc"][m], bins=np.logspace(1, np.log10(125), 21), **kw)
        ax[0, 1].hist(cos_t[m], bins=np.linspace(-0.8, 0.8, 33), **kw)
        ax[0, 2].hist(d["phi"][m], bins=np.linspace(-180, 180, 73), **kw)
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel(r"$E_\mathrm{mc}$ [GeV]")
    ax[0, 1].set_xlabel(r"$\cos\theta$")
    ax[0, 2].set_xlabel(r"$\phi$ [deg]")
    for a in ax[0]:
        a.set_ylabel("unit-area density")
    ax[0, 0].legend(frameon=False, fontsize=10)
    ax[0, 0].set_title("Do the three groups cover the same phase space?\n(each curve normalised to unit area)", fontsize=11)
    cc3 = np.isin(grp, (1, 2))
    fb = grp == 2
    for a, x, edges, xl in ((ax[1, 0], d["E_mc"], E_BINS, r"$E_\mathrm{mc}$ [GeV]"),
                            (ax[1, 1], np.abs(cos_t), np.linspace(0, 0.8, 9), r"$|\cos\theta|$"),
                            (ax[1, 2], d["phi_crack_dist"], PHI_DIST_BINS, "signed φ-distance to nearest crack [deg]")):
        c, p, e, _ = _rate(x, fb, cc3, edges)
        a.errorbar(c, p, yerr=e, color=GROUP_COLOR[2], marker="^", capsize=3, lw=2)
        a.set_xlabel(xl)
        a.set_ylabel("fallback / (True-CC3 + fallback)")
        a.set_ylim(bottom=0)
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_title("Fallback share (CC3 file); binomial errors, N>=5 per bin", fontsize=11)
    fig.suptitle(NOTE, fontsize=10, color="#555555")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figA_population_check.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for i, (t, f) in enumerate(FLOORS):
        for j, (cond, ttl) in enumerate(((d["cat_" + t] != 1, "not owned by a photon PFO (cat 2-6)"),
                                         (d["cat_" + t] == 6, "nothing linked in photon/descendants/ancestors (cat 6)"))):
            for g in (0, 1, 2):
                c, p, e, _ = _rate(d["E_mc"], cond, grp == g, E_BINS)
                if len(c):
                    ax[i, j].errorbar(c, p, yerr=e, color=GROUP_COLOR[g], marker="os^"[g], capsize=3,
                                      lw=2, label=GROUP_LABEL[g])
            ax[i, j].set_title(f"floor {f}: {ttl}", fontsize=10)
            ax[i, j].set_ylabel("fraction of all truth photons in group")
            ax[i, j].set_ylim(bottom=0)
    for a in ax[1]:
        a.set_xscale("log")
        a.set_xlabel(r"$E_\mathrm{mc}$ [GeV]")
    ax[0, 0].legend(frameon=False, fontsize=10)
    fig.suptitle("Lost fraction vs energy (denominator: all truth photons of the group in the bin)\n" + NOTE, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figB_lost_fraction.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    cols = ["#1b9e77", "#66a61e", "#e6ab02", "#d95f02", "#7570b3", "#999999"]
    cnames = ["1 owned", "2 shared", "3 non-photon PFO", "4 via descendants", "5 via ancestors", "6 none"]
    for i, (t, f) in enumerate(FLOORS):
        bottom = np.zeros(3)
        for c in range(1, 7):
            vals = np.array([((grp == g) & (d["cat_" + t] == c)).sum() / max((grp == g).sum(), 1) for g in (0, 1, 2)])
            ax[i].bar(range(3), vals, bottom=bottom, color=cols[c - 1], label=cnames[c - 1])
            bottom += vals
        ax[i].set_xticks(range(3))
        ax[i].set_xticklabels([GROUP_LABEL[g] for g in (0, 1, 2)], fontsize=10)
        ax[i].set_title(f"floor {f}")
    ax[0].set_ylabel("share of all truth photons in group")
    ax[1].legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("How is each photon's energy reconstructed? (link category, priority order)\n" + NOTE, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figC_link_categories.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for i, (t, f) in enumerate(FLOORS):
        for j, (key, ttl) in enumerate((("comp_own_", "own links only"), ("comp_fam_", "incl. descendants"))):
            for g in (0, 1, 2):
                m = grp == g
                if m.sum() >= MIN_PLOT_N:
                    ax[i, j].hist(np.clip(d[key + t][m], 0, 1.5), bins=np.linspace(0, 1.5, 61),
                                  histtype="step", density=True, color=GROUP_COLOR[g], lw=2, label=GROUP_LABEL[g])
            ax[i, j].set_yscale("log")
            ax[i, j].set_title(f"floor {f}: completeness, {ttl}", fontsize=10)
            ax[i, j].set_xlabel(r"$\sum cw\,E_\mathrm{pfo}\,/\,E_\mathrm{mc}$ (clipped to 1.5)")
            ax[i, j].set_ylabel("unit-area density")
    ax[0, 0].legend(frameon=False, fontsize=10)
    fig.suptitle("Truth energy recovered through linked PFOs (log y; spike at 0 = nothing recovered)\n" + NOTE, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figD_completeness.png"), bbox_inches="tight")
    plt.close(fig)
    log("  saved figA_population_check.png, figB_lost_fraction.png, figC_link_categories.png, figD_completeness.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["1k", "10k", "100k"], default="1k")
    ap.add_argument("--group", default="")
    ap.add_argument("--ecal-r", type=float, default=1800.0)
    args = ap.parse_args()
    rec = "/data/dust/user/alimuham/thesis/reco/"
    paths = {
        "1k": (rec + "tau-pi0-g4-taureco-1000events-test_REC.edm4hep.root",
               rec + "tau-pi0-cc3-taureco-1000events-test_REC.edm4hep.root", 1000),
        "10k": (rec + "tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root",
                rec + "tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root", 10000),
        "100k": (rec + "tau-pi0-geant4-100kevents-reco-test_REC.edm4hep.root",
                 rec + "tau-pi0-caloclouds-100kevents-reco-test_REC.edm4hep.root", 100000),
    }
    g4f, cc3f, nmax = paths[args.dataset]
    for f in (g4f, cc3f):
        if not os.path.exists(f):
            sys.exit(f"Input file not found: {f}")
    out = os.path.join(os.path.expanduser("~/thesis-ml-sim/plots"), args.group,
                       f"fallback_study_v1_{args.dataset}")
    os.makedirs(out, exist_ok=True)
    log(f"fallback_study_v1.py   dataset {args.dataset}   output {out}")
    rows = run(g4f, "Geant4 reco file", False, nmax) + run(cc3f, "CC3 reco file", True, nmax)
    d = to_arrays(rows)
    if not d:
        sys.exit("No truth photons selected.")
    np.savez_compressed(os.path.join(out, f"fallback_photons_{args.dataset}.npz"), **d)
    log(f"saved fallback_photons_{args.dataset}.npz ({len(rows)} rows, {len(d)} columns)")
    d["_ecal_r"] = args.ecal_r
    summarize(d)
    make_figures(d, out)
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    log(f"\nDone. Output: {out}")


if __name__ == "__main__":
    main()
