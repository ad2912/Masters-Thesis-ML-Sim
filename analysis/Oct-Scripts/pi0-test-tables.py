#!/usr/bin/env python3
"""
analyse_pi0_tables.py -- reads the two NPZ files written by make_pi0_tables.py (G4 and CC3) and answers ONE question:
   WHERE in the chain does CC3 end up with fewer true pi0 pairs than G4?

RUN WITH NO ARGUMENTS (full 100k files).  Test on the 1k files:  python analyse_pi0_tables.py --suffix first1000

THE CHAIN (every true pi0 ends in exactly one outcome; reconstruction outcomes first, finder outcomes after):
  RECONSTRUCTION  neither      : neither photon has a matching PFO
                  one_missing  : only one photon has a matching PFO
                  merged       : both photons match the SAME PFO
                  other_type   : both match different PFOs, but at least one is not a photon (PDG 22) PFO
  FINDER          res_nf_energy: two separate photon PFOs ("resolved"), a PFO has E < 0.5 GeV, no candidate
                  res_nf_window: resolved, energies fine, unfitted pair mass outside 95-175 MeV, no candidate
                  res_nf_other : resolved, passes both cuts, but the finder made no candidate (fit probability? unknown)
  FOUND           res_found    : the finder made a candidate from exactly these two PFOs
  "matching PFO" of a photon = its best PFO (rank = cw * E_pfo) with cluster weight cw >= floor (floors 0.5 and 0.1).

POPULATIONS:  gen = both photons genStatus 1 (generator pi0s: PRIMARY, identical in both files by construction)
              all = every two-photon pi0 in the table (adds simulation-created pi0s; closer to the old cache)
BLOCKS:  1 sanity | 2 A1 check | 3 closure vs old cache | 4 outcomes | 5 cut ladder | 6 reco-side build | 7 separation | 8 readout
STATISTICS: fractions: binomial error.  CC3-G4 differences in pi0 counts and ratios: independent-Poisson error.
  G4 and CC3 share generator events, so these errors are APPROXIMATE (usually conservative). No systematics.
  No Geant4-vs-Geant4 baseline exists: the size of random scatter between two full simulations is unknown.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

REC_DIR = "/data/dust/user/alimuham/thesis/reco"
E_EDGES = [5.0, 10.0, 20.0, 30.0, 50.0, 125.0]            # inner edges, lower edge inclusive
E_NAMES = ["<5", "5-10", "10-20", "20-30", "30-50", "50-125", ">=125"]
SEP_EDGES = [5.0, 10.0, 20.0, 40.0]
SEP_NAMES = ["0-5", "5-10", "10-20", "20-40", ">=40"]
E_CUT = 0.5                                                # GeV, finder photon cut (E assumed; |p| also checked)
M_LO, M_HI = 0.1349766 - 0.04, 0.1349766 + 0.04            # GeV, finder window around the pi0 mass
R_ECAL_MM = 1843.0                                         # ILD baseline barrel ECAL inner radius
BARREL_THETA = (40.0, 140.0)                               # degrees, used only for the separation split
FLOOR_EPS = 1e-6
OUTCOMES = ["neither", "one_missing", "merged", "other_type",
            "res_nf_energy", "res_nf_window", "res_nf_other", "res_found"]
STAGE = ["reco", "reco", "reco", "reco", "finder", "finder", "finder", "found"]
PRIMARY = ("gen", 0.5)
POPULATIONS = ("gen", "all")
FLOORS = (0.5, 0.1)

# numbers printed by the old cache scripts (100k files only), for the closure test
CACHE_EXPECT = {
    ("G4", 0.5): dict(no_photon=3802, one_photon=22766, two_same=498, two_distinct=84609, same_pi0=63491, unique_pi0=63424),
    ("CC3", 0.5): dict(no_photon=4119, one_photon=23359, two_same=610, two_distinct=81604, same_pi0=60506, unique_pi0=60429),
    ("G4", 0.1): dict(no_photon=3261, one_photon=21998, two_same=619, two_distinct=85797, same_pi0=63664),
    ("CC3", 0.1): dict(no_photon=3503, one_photon=22507, two_same=752, two_distinct=82930, same_pi0=60696),
}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
class Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "w"), sys.__stdout__

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


def bin_index(x, edges):
    x = np.asarray(x, dtype=float)
    b = np.digitize(x, edges)
    b[~np.isfinite(x)] = -1
    return b


def binom_err(k, n):
    return math.sqrt(max(k / n * (1 - k / n), 0.0) / n) if n > 0 else float("nan")


def ratio_err(a, b):
    return a / b * math.sqrt(1.0 / a + 1.0 / b) if a > 0 and b > 0 else float("nan")


def pair_key(loop, a, b):
    loop, a, b = (np.asarray(v, dtype=np.int64) for v in (loop, a, b))
    if a.size and (max(a.max(), b.max()) >= 2 ** 21):
        raise ValueError("PFO index too large for the key packing")
    return (loop << 42) | (np.minimum(a, b) << 21) | np.maximum(a, b)


def sep_mm(open_angle_rad, theta_deg):
    """Separation of the two photons on the BARREL ECAL face: angle * R / sin(theta). Small angle, octagon ignored."""
    return np.asarray(open_angle_rad) * R_ECAL_MM / np.sin(np.radians(np.asarray(theta_deg, dtype=float)))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class Data:
    def __init__(self, path, tag):
        z = np.load(path)
        self.tag, self.path = tag, path
        self.a = {k: z[k] for k in z.files if not k.startswith("meta_")}
        self.meta = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}

    def __getitem__(self, k):
        return self.a[k]


def restrict_common(d1, d2):
    def rk(r, e):
        return (np.asarray(r, dtype=np.int64) << 32) | np.asarray(e, dtype=np.int64)
    common = np.intersect1d(rk(d1["ev_run"], d1["ev_evt"]), rk(d2["ev_run"], d2["ev_evt"]))
    for d in (d1, d2):
        for pre in ("ev_", "mc_", "pfo_", "cand_"):
            m = np.isin(rk(d[pre + "run"], d[pre + "evt"]), common)
            for k in [k for k in d.a if k.startswith(pre)]:
                d.a[k] = d.a[k][m]
    return len(common)


def population_mask(d, pop):
    n = len(d["mc_pi0_idx"])
    if pop == "gen":
        return (d["mc_ph0_gen"] == 1) & (d["mc_ph1_gen"] == 1)
    return np.ones(n, dtype=bool)


# ----------------------------------------------------------------------------------------------
# core: outcome of every true pi0
# ----------------------------------------------------------------------------------------------
def candidate_keys(d):
    ok = (d["cand_d0_pfo_idx"] >= 0) & (d["cand_d1_pfo_idx"] >= 0)
    return pair_key(d["cand_loop_evt"][ok], d["cand_d0_pfo_idx"][ok], d["cand_d1_pfo_idx"][ok])


def mc_outcomes(d, floor, cand_keys):
    n = len(d["mc_pi0_idx"])
    ia, ib = d["mc_ph0_b0_pfo_idx"].astype(np.int64), d["mc_ph1_b0_pfo_idx"].astype(np.int64)
    ma = (ia >= 0) & (d["mc_ph0_b0_cw"].astype(float) >= floor - FLOOR_EPS)
    mb = (ib >= 0) & (d["mc_ph1_b0_cw"].astype(float) >= floor - FLOOR_EPS)
    nm = ma.astype(int) + mb.astype(int)
    code = np.full(n, -1, dtype=np.int8)
    code[nm == 0] = 0
    code[nm == 1] = 1
    both = nm == 2
    merged = both & (ia == ib)
    code[merged] = 2
    other = both & ~merged & ((d["mc_ph0_b0_pdg"] != 22) | (d["mc_ph1_b0_pdg"] != 22))
    code[other] = 3
    res = both & ~merged & ~other
    with np.errstate(invalid="ignore"):
        e1, e2 = d["mc_ph0_b0_E"], d["mc_ph1_b0_E"]
        px = d["mc_ph0_b0_px"] + d["mc_ph1_b0_px"]
        py = d["mc_ph0_b0_py"] + d["mc_ph1_b0_py"]
        pz = d["mc_ph0_b0_pz"] + d["mc_ph1_b0_pz"]
        m2 = (e1 + e2) ** 2 - (px ** 2 + py ** 2 + pz ** 2)
        mass = np.sqrt(np.clip(m2, 0.0, None))
        p1 = np.sqrt(d["mc_ph0_b0_px"] ** 2 + d["mc_ph0_b0_py"] ** 2 + d["mc_ph0_b0_pz"] ** 2)
        p2 = np.sqrt(d["mc_ph1_b0_px"] ** 2 + d["mc_ph1_b0_py"] ** 2 + d["mc_ph1_b0_pz"] ** 2)
        E_ok = res & (e1 >= E_CUT) & (e2 >= E_CUT)
        p_ok = res & (p1 >= E_CUT) & (p2 >= E_CUT)
        M_ok = res & (mass >= M_LO) & (mass <= M_HI)
    found = np.zeros(n, dtype=bool)
    idx = np.where(res)[0]
    if idx.size:
        found[idx] = np.isin(pair_key(d["mc_loop_evt"][idx], ia[idx], ib[idx]), cand_keys)
    code[res & found] = 7
    code[res & ~found & ~E_ok] = 4
    code[res & ~found & E_ok & ~M_ok] = 5
    code[res & ~found & E_ok & M_ok] = 6
    diag = dict(resolved=int(res.sum()), negative_m2=int((res & (m2 < 0)).sum()),
                found_but_fails_E_cut=int((found & ~E_ok).sum()), found_but_fails_window=int((found & ~M_ok).sum()),
                E_cut_vs_p_cut_disagree=int((res & (E_ok != p_ok)).sum()))
    return dict(code=code, res=res, E_ok=E_ok, M_ok=M_ok, found=found, mass=mass, diag=diag)


def count_matrix(code, b, mask, nbins):
    M = np.zeros((nbins + 1, len(OUTCOMES)), dtype=np.int64)
    ok = mask & (b >= 0) & (code >= 0)
    np.add.at(M, (b[ok], code[ok]), 1)
    M[-1] = M[:-1].sum(axis=0)
    return M


# ----------------------------------------------------------------------------------------------
# closure against the old candidate cache (same definitions as the old plots)
# ----------------------------------------------------------------------------------------------
def cache_classes(d, floor):
    pk = (d["pfo_loop_evt"].astype(np.int64) << 21) | d["pfo_pfo_idx"].astype(np.int64)
    order = np.argsort(pk, kind="stable")
    pks = pk[order]
    cw = d["pfo_l0_cw"].astype(float)
    res = {}

    def lookup(col):
        key = (d["cand_loop_evt"].astype(np.int64) << 21) | np.maximum(d[col], 0).astype(np.int64)
        pos = np.minimum(np.searchsorted(pks, key), len(pks) - 1)
        ok = (d[col] >= 0) & (pks[pos] == key)
        return order[pos], ok

    r0, ok0 = lookup("cand_d0_pfo_idx")
    r1, ok1 = lookup("cand_d1_pfo_idx")
    res["cand_total"] = len(ok0)
    res["daughter_missing_in_pfo_table"] = int((~ok0).sum() + (~ok1).sum())
    ph0 = ok0 & (cw[r0] >= floor - FLOOR_EPS) & (d["pfo_l0_pdg"][r0] == 22)
    ph1 = ok1 & (cw[r1] >= floor - FLOOR_EPS) & (d["pfo_l0_pdg"][r1] == 22)
    nph = ph0.astype(int) + ph1.astype(int)
    both = nph == 2
    same_ph = both & (d["pfo_l0_idx"][r0] == d["pfo_l0_idx"][r1])
    dist = both & ~same_ph
    par0, par1 = d["pfo_l0_pi0_par_idx"][r0], d["pfo_l0_pi0_par_idx"][r1]
    same_pi0 = dist & (par0 >= 0) & (par0 == par1)
    res.update(no_photon=int((nph == 0).sum()), one_photon=int((nph == 1).sum()), two_same=int(same_ph.sum()),
               two_distinct=int(dist.sum()), same_pi0=int(same_pi0.sum()))
    ukey = (d["cand_loop_evt"].astype(np.int64)[same_pi0] << 22) | par0[same_pi0].astype(np.int64)
    res["unique_pi0"] = int(np.unique(ukey).size)
    return res


# ----------------------------------------------------------------------------------------------
# reco-side build: group photon PFOs by the true pi0 of their top-link photon
# ----------------------------------------------------------------------------------------------
def reco_side_keys(d, floor):
    sel = ((d["pfo_l0_cw"].astype(float) >= floor - FLOOR_EPS) & (d["pfo_l0_pdg"] == 22)
           & (d["pfo_l0_pi0_par_idx"] >= 0))
    loop = d["pfo_loop_evt"][sel].astype(np.int64)
    par = d["pfo_l0_pi0_par_idx"][sel].astype(np.int64)
    ph = d["pfo_l0_idx"][sel].astype(np.int64)
    if par.size and (max(par.max(), ph.max()) >= 2 ** 22):
        raise ValueError("MC index too large for key packing")
    trip = np.unique((((loop << 22) | par) << 22) | ph)
    uk, cnt = np.unique(trip >> 22, return_counts=True)
    return uk[cnt >= 2]


def mc_row_key(d):
    return (d["mc_loop_evt"].astype(np.int64) << 22) | d["mc_pi0_idx"].astype(np.int64)


# ----------------------------------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------------------------------
def print_outcome_table(title, names, MG, MC, only_all=False):
    print("\n" + title)
    print("   error on fractions: binomial; on CC3-G4 [pi0s]: sqrt(nG+nC) (independent Poisson, approximate); z = diff/err")
    rows = []
    sel = [len(names)] if only_all else list(range(len(names) + 1))
    for b in sel:
        bn = "all" if b == len(names) else names[b]
        NG, NC = int(MG[b].sum()), int(MC[b].sum())
        print("  -- %s : N_true G4 = %d, CC3 = %d (CC3-G4 = %+d)" % (bn, NG, NC, NC - NG))
        print("     %-14s %-7s %18s %18s %20s %7s" % ("outcome", "stage", "G4 n (frac %)", "CC3 n (frac %)", "CC3-G4 [pi0s]", "z"))
        for j, o in enumerate(OUTCOMES):
            nG, nC = int(MG[b, j]), int(MC[b, j])
            fG, fC = (nG / NG if NG else float("nan")), (nC / NC if NC else float("nan"))
            eG, eC = binom_err(nG, NG), binom_err(nC, NC)
            diff, err = nC - nG, math.sqrt(nG + nC)
            z = diff / err if err > 0 else float("nan")
            print("     %-14s %-7s %9d (%5.1f+-%.1f) %9d (%5.1f+-%.1f) %+10d +- %-6.0f %+7.1f" %
                  (o, STAGE[j], nG, 100 * fG, 100 * eG, nC, 100 * fC, 100 * eC, diff, err, z))
            rows.append([bn, o, STAGE[j], NG, NC, nG, nC, fG, fC, eG, eC, diff, err, z])
        print("     (sum of the CC3-G4 column = %+d = N_true difference)" % int((MC[b] - MG[b]).sum()))
    return rows


OUT_HEADER = ["bin", "outcome", "stage", "N_true_G4", "N_true_CC3", "n_G4", "n_CC3", "frac_G4", "frac_CC3",
              "frac_err_G4", "frac_err_CC3", "diff_n", "diff_err", "z"]


def ladder_counts(o, b, mask, nbins):
    out = {}
    for name, m in (("N_true", np.ones_like(o["res"])), ("resolved", o["res"]), ("res+E", o["E_ok"]),
                    ("res+E+window", o["E_ok"] & o["M_ok"]), ("found", o["found"])):
        M = np.zeros(nbins + 1, dtype=np.int64)
        ok = mask & (b >= 0) & m
        np.add.at(M, b[ok], 1)
        M[-1] = M[:-1].sum()
        out[name] = M
    return out


def print_ladder(title, names, LG, LC):
    print("\n" + title)
    print("   steps are cumulative: resolved -> + both PFO E >= 0.5 GeV -> + unfitted mass in 95-175 MeV; 'found' = real finder")
    print("   ratio = CC3/G4 (independent-Poisson error, approximate). pass rate = step / previous step, binomial error.")
    rows = []
    for b in range(len(names) + 1):
        bn = "all" if b == len(names) else names[b]
        print("  -- %s" % bn)
        print("     %-14s %9s %9s %18s %24s" % ("step", "G4", "CC3", "CC3/G4", "pass rate G4 | CC3 [%]"))
        prev = None
        for s in ("N_true", "resolved", "res+E", "res+E+window", "found"):
            nG, nC = int(LG[s][b]), int(LC[s][b])
            r = ("%.3f +- %.3f" % (nC / nG, ratio_err(nC, nG))) if nG > 0 and nC > 0 else "n/a"
            pr = ""
            if prev is not None and prev[0] > 0 and prev[1] > 0:
                pr = "%5.1f+-%.1f | %5.1f+-%.1f" % (100 * nG / prev[0], 100 * binom_err(nG, prev[0]),
                                                     100 * nC / prev[1], 100 * binom_err(nC, prev[1]))
            print("     %-14s %9d %9d %18s %24s" % (s, nG, nC, r, pr))
            rows.append([bn, s, nG, nC, (nC / nG if nG else float("nan"))])
            prev = (nG, nC)
        # found as a share of resolved, directly
        nG, nC = int(LG["found"][b]), int(LC["found"][b])
        rG, rC = int(LG["resolved"][b]), int(LC["resolved"][b])
        if rG > 0 and rC > 0:
            print("     found / resolved : G4 %.1f+-%.1f %%, CC3 %.1f+-%.1f %%" %
                  (100 * nG / rG, 100 * binom_err(nG, rG), 100 * nC / rC, 100 * binom_err(nC, rC)))
    return rows


# ----------------------------------------------------------------------------------------------
# the analysis
# ----------------------------------------------------------------------------------------------
def run_analysis(path_g4, path_cc3, outdir, common_only=False):
    os.makedirs(outdir, exist_ok=True)
    tee = Tee(os.path.join(outdir, "summary.txt"))
    sys.stdout = tee
    try:
        return _run(path_g4, path_cc3, outdir, common_only)
    finally:
        sys.stdout = sys.__stdout__
        tee.f.close()


def _run(path_g4, path_cc3, outdir, common_only):
    G, C = Data(path_g4, "G4"), Data(path_cc3, "CC3")
    result = {}
    print("=" * 100)
    print("MAP:  1 sanity | 2 A1 check | 3 closure vs old cache | 4 outcomes | 5 cut ladder | 6 reco-side build | "
          "7 separation | 8 readout")
    print("G4 :", G.path, "\nCC3:", C.path, "\noutput:", outdir)
    print("=" * 100)

    # ---- 1. sanity
    print("\n[1] SANITY")
    if common_only:
        print("   --common-only: restricted to %d events present (run, evt) in both files" % restrict_common(G, C))
    for d in (G, C):
        ev_key = (d["ev_run"].astype(np.int64) << 32) | d["ev_evt"].astype(np.int64)
        gens = {}
        for a, b in zip(d["mc_ph0_gen"], d["mc_ph1_gen"]):
            gens[(int(a), int(b))] = gens.get((int(a), int(b)), 0) + 1
        print("  %s: events %d (unique run/evt %d) | mc rows %d | photon-PFO rows %d | candidates %d" %
              (d.tag, len(d["ev_loop"]), np.unique(ev_key).size, len(d["mc_pi0_idx"]), len(d["pfo_pfo_idx"]),
               len(d["cand_cand_idx"])))
        print("      photon gen status pairs (ph0, ph1): %s | NaN pi0_E: %d | NaN true_open_angle: %d" %
              (gens, int(np.isnan(d["mc_pi0_E"]).sum()), int(np.isnan(d["mc_true_open_angle"]).sum())))
        keep = ("events_read", "links_total", "links_duplicate_dropped", "links_bad_rec_index", "links_bad_sim_index",
                "cand_not_2_daughters", "cand_daughter_not_in_pfo_collection", "pfo_index_not_position",
                "pi0_total", "pi0_rows", "pi0_skip_not_two_daughters")
        print("      maker counters:", {k: int(d.meta[k]) for k in keep if k in d.meta})
    kG = (G["ev_run"].astype(np.int64) << 32) | G["ev_evt"].astype(np.int64)
    kC = (C["ev_run"].astype(np.int64) << 32) | C["ev_evt"].astype(np.int64)
    print("  event numbers present in both files: %d (G4 rows %d, CC3 rows %d)" %
          (np.intersect1d(kG, kC).size, len(kG), len(kC)))
    n_events = min(len(G["ev_loop"]), len(C["ev_loop"]))
    full = n_events >= 99000

    # ---- 2. A1
    print("\n[2] A1 CHECK: are the true pi0s the same in both files?  (true pi0 energy bins, rows of the mc table)")
    for pop in POPULATIONS:
        mG, mC = population_mask(G, pop), population_mask(C, pop)
        bG, bC = bin_index(G["mc_pi0_E"], E_EDGES), bin_index(C["mc_pi0_E"], E_EDGES)
        MG = np.zeros(len(E_NAMES) + 1, dtype=np.int64)
        MC = np.zeros(len(E_NAMES) + 1, dtype=np.int64)
        for M, b, m in ((MG, bG, mG), (MC, bC, mC)):
            ok = m & (b >= 0)
            np.add.at(M, b[ok], 1)
            M[-1] = M[:-1].sum()
        print("  population %-3s : " % pop + " | ".join("%s: %d/%d" % (("all" if i == len(E_NAMES) else E_NAMES[i]),
                                                                       MG[i], MC[i]) for i in range(len(E_NAMES) + 1)))
        print("                    (G4/CC3 counts per bin; identical = generator-level truth is the same)")
        result["A1_" + pop] = (MG, MC)

    # ---- 3. closure
    print("\n[3] CLOSURE: do the new tables reproduce the old cache numbers? (candidates, same definitions as the old plots)")
    for d in (G, C):
        for fl in FLOORS:
            r = cache_classes(d, fl)
            result[("closure", d.tag, fl)] = r
            exp = CACHE_EXPECT.get((d.tag, fl))
            print("  %s floor %.1f : %s" % (d.tag, fl, {k: r[k] for k in r}))
            if exp and full:
                bad = {k: (r[k], v) for k, v in exp.items() if r[k] != v}
                print("      vs old cache (new, old): %s" % ("ALL EQUAL" if not bad else bad))
            elif exp:
                print("      (old-cache numbers are for the full 100k files only: not compared)")

    # ---- 4/5. outcomes and ladder
    ck_G, ck_C = candidate_keys(G), candidate_keys(C)
    eb_G, eb_C = bin_index(G["mc_pi0_E"], E_EDGES), bin_index(C["mc_pi0_E"], E_EDGES)
    csv_rows, ladder_rows, outcomes = [], [], {}
    for pop in POPULATIONS:
        for fl in FLOORS:
            oG, oC = mc_outcomes(G, fl, ck_G), mc_outcomes(C, fl, ck_C)
            outcomes[(pop, fl)] = (oG, oC)
            mG, mC = population_mask(G, pop), population_mask(C, pop)
            MG = count_matrix(oG["code"], eb_G, mG, len(E_NAMES))
            MC = count_matrix(oC["code"], eb_C, mC, len(E_NAMES))
            result[("outcomes", pop, fl)] = (MG, MC)
            primary = (pop, fl) == PRIMARY
            title = "[4] OUTCOMES  population=%s  weight floor=%.1f  (bins: true pi0 energy [GeV]; denominator = true pi0s in the bin)" % (pop, fl)
            rows = print_outcome_table(title, E_NAMES, MG, MC, only_all=not primary)
            csv_rows += [[pop, fl] + r for r in rows]
            print("   emulation diagnostics (resolved pairs only) G4: %s" % oG["diag"])
            print("                                              CC3: %s" % oC["diag"])
            LG = ladder_counts(oG, eb_G, mG, len(E_NAMES))
            LC = ladder_counts(oC, eb_C, mC, len(E_NAMES))
            result[("ladder", pop, fl)] = (LG, LC)
            sub = E_NAMES if primary else E_NAMES
            rows = print_ladder("[5] CUT LADDER  population=%s  floor=%.1f" % (pop, fl), E_NAMES, LG, LC) if primary else []
            if not primary:
                print("\n[5] CUT LADDER population=%s floor=%.1f : all-energy line only" % (pop, fl))
                for s in ("N_true", "resolved", "res+E", "res+E+window", "found"):
                    nG, nC = int(LG[s][-1]), int(LC[s][-1])
                    print("     %-14s G4 %9d  CC3 %9d  CC3/G4 %s" % (s, nG, nC,
                          ("%.3f +- %.3f" % (nC / nG, ratio_err(nC, nG))) if nG > 0 and nC > 0 else "n/a"))
            ladder_rows += [[pop, fl] + r for r in rows]
            for b in range(len(E_NAMES) + 1):
                for s in ("N_true", "resolved", "res+E", "res+E+window", "found"):
                    if not primary:
                        ladder_rows.append([pop, fl, "all" if b == len(E_NAMES) else E_NAMES[b], s,
                                            int(LG[s][b]), int(LC[s][b]),
                                            (LC[s][b] / LG[s][b] if LG[s][b] else float("nan"))]) if b == len(E_NAMES) else None
    write_csv(os.path.join(outdir, "outcomes.csv"), ["population", "floor"] + OUT_HEADER, csv_rows)
    write_csv(os.path.join(outdir, "ladder.csv"), ["population", "floor", "bin", "step", "G4", "CC3", "CC3_over_G4"], ladder_rows)

    # ---- 6. reco-side build
    print("\n[6] RECO-SIDE BUILD: group photon PFOs by the true pi0 of their top-link photon (no cuts of any kind)")
    print("   A pi0 counts when >= 2 PFOs have top links (cw >= floor, photon PDG 22) to DIFFERENT photons of that pi0.")
    print("   Compared with the MC-side 'resolved' (best PFO of each photon). Differences = definition sensitivity.")
    for fl in FLOORS:
        kG, kC = reco_side_keys(G, fl), reco_side_keys(C, fl)
        for pop in POPULATIONS if fl == PRIMARY[1] else ("gen",):
            oG, oC = outcomes[(pop, fl)]
            mG, mC = population_mask(G, pop), population_mask(C, pop)
            rsG = np.isin(mc_row_key(G), kG) & mG
            rsC = np.isin(mc_row_key(C), kC) & mC
            msG, msC = oG["res"] & mG, oC["res"] & mC
            print("  population=%s floor=%.1f" % (pop, fl))
            print("     %-8s %12s %12s %16s %12s %12s %16s" % ("bin", "reco G4", "reco CC3", "CC3/G4", "MC G4", "MC CC3", "CC3/G4"))
            for b in list(range(len(E_NAMES))) + [None]:
                if b is None:
                    sG, sC, tG, tC = rsG.sum(), rsC.sum(), msG.sum(), msC.sum()
                else:
                    sG, sC = (rsG & (eb_G == b)).sum(), (rsC & (eb_C == b)).sum()
                    tG, tC = (msG & (eb_G == b)).sum(), (msC & (eb_C == b)).sum()
                f = lambda a, c: ("%.3f+-%.3f" % (c / a, ratio_err(c, a))) if a > 0 and c > 0 else "n/a"
                print("     %-8s %12d %12d %16s %12d %12d %16s" % ("all" if b is None else E_NAMES[b], sG, sC, f(sG, sC), tG, tC, f(tG, tC)))
            print("     overlap (both / reco-only / MC-only):  G4 %d / %d / %d    CC3 %d / %d / %d" %
                  ((rsG & msG).sum(), (rsG & ~msG).sum(), (~rsG & msG).sum(),
                   (rsC & msC).sum(), (rsC & ~msC).sum(), (~rsC & msC).sum()))
            result[("reco_side", pop, fl)] = (int(rsG.sum()), int(rsC.sum()), int(msG.sum()), int(msC.sum()))
        print("     reco-side pi0 keys not found in the mc table: G4 %d, CC3 %d (e.g. pi0s with a Dalitz decay)" %
              (kG.size - np.isin(kG, mc_row_key(G)).sum(), kC.size - np.isin(kC, mc_row_key(C)).sum()))

    # ---- 7. separation
    print("\n[7] SEPARATION SPLIT (primary population/floor). separation = true opening angle * %.0f mm / sin(theta_pi0)" % R_ECAL_MM)
    print("   BARREL pi0s only (pi0 theta %.0f-%.0f deg); the formula is wrong in the endcap. Octagon corners (up to 8%%) ignored." % BARREL_THETA)
    pop, fl = PRIMARY
    oG, oC = outcomes[PRIMARY]
    sepG = sep_mm(G["mc_true_open_angle"], G["mc_pi0_theta"])
    sepC = sep_mm(C["mc_true_open_angle"], C["mc_pi0_theta"])
    bar = lambda d: (d["mc_pi0_theta"] >= BARREL_THETA[0]) & (d["mc_pi0_theta"] <= BARREL_THETA[1])
    sbG, sbC = bin_index(sepG, SEP_EDGES), bin_index(sepC, SEP_EDGES)
    mG, mC = population_mask(G, pop) & bar(G), population_mask(C, pop) & bar(C)
    MG = count_matrix(oG["code"], sbG, mG, len(SEP_NAMES))
    MC = count_matrix(oC["code"], sbC, mC, len(SEP_NAMES))
    result["sep"] = (MG, MC)
    rows = print_outcome_table("   bins: separation [mm]", SEP_NAMES, MG, MC)
    write_csv(os.path.join(outdir, "separation.csv"), OUT_HEADER, rows)
    rows2 = []
    for eg in range(len(E_NAMES)):
        for sb in range(len(SEP_NAMES)):
            a = (mG & (eb_G == eg) & (sbG == sb))
            c = (mC & (eb_C == eg) & (sbC == sb))
            r_ = lambda m, o: int((m & o["res"]).sum())
            f_ = lambda m, o: int((m & o["found"]).sum())
            rows2.append([E_NAMES[eg], SEP_NAMES[sb], int(a.sum()), int(c.sum()), r_(a, oG), r_(c, oC), f_(a, oG), f_(c, oC)])
    write_csv(os.path.join(outdir, "separation_by_energy.csv"),
              ["E_bin", "sep_bin", "N_G4", "N_CC3", "resolved_G4", "resolved_CC3", "found_G4", "found_CC3"], rows2)

    # ---- 8. readout
    print("\n[8] READOUT (primary population/floor): the two largest CC3-G4 changes per true-energy bin")
    print("   Reading rule: 'reco' outcomes = photons missing/merged BEFORE the finder; 'finder' outcomes = lost AT the finder.")
    MG, MC = result[("outcomes",) + PRIMARY]
    for b in range(len(E_NAMES) + 1):
        diff = MC[b] - MG[b]
        err = np.sqrt(MC[b] + MG[b]).astype(float)
        z = np.divide(diff, err, out=np.zeros_like(err), where=err > 0)
        top = np.argsort(-np.abs(z))[:2]
        print("   %-7s N_true %d/%d : " % ("all" if b == len(E_NAMES) else E_NAMES[b], MG[b].sum(), MC[b].sum()) +
              "; ".join("%s %+d (z=%+.1f, %s)" % (OUTCOMES[j], diff[j], z[j], STAGE[j]) for j in top))
    print("\nWHAT THIS CANNOT TELL YOU: why a stage changes (hits, Pandora behaviour, finder response); no Geant4-vs-Geant4")
    print("baseline; errors are approximate (shared generator events); selection is 'two-photon pi0s with exactly 2 daughters'.")
    print("\nFiles written to", outdir)
    return result


def main():
    ap = argparse.ArgumentParser(description="Run with no arguments: full 100k NPZ files.")
    ap.add_argument("--suffix", default="full", help="full (default) or first1000")
    ap.add_argument("--indir", default=REC_DIR)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--common-only", action="store_true", help="keep only events (run, evt) present in both files")
    a = ap.parse_args()
    g4 = os.path.join(a.indir, "pi0-tables-g4-%s.npz" % a.suffix)
    cc3 = os.path.join(a.indir, "pi0-tables-cc3-%s.npz" % a.suffix)
    out = a.outdir or os.path.join(a.indir, "analysis-%s" % a.suffix)
    run_analysis(g4, cc3, out, a.common_only)


if __name__ == "__main__":
    sys.exit(main())
