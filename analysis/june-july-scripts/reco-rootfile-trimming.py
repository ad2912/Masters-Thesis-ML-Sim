"""
slim_reco_files.py -- Slim G4 and CC3 EDM4hep reco files for Computer upload.

Run on NAF:
    source ~/source.sh
    python3 slim_reco_files.py
"""

import ROOT
import os

# CONFIG
MAX_EVENTS = 100  # set to None to keep all events, just drop branches

FILES = {
    "g4": {
        "in":  "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root",
        "out": "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-slim_REC.edm4hep.root",
    },
    "cc3": {
        "in":  "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root",
        "out": "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-slim_REC.edm4hep.root",
    },
}

DROP_PATTERNS = [
    "EcalBarrelCollection", "EcalEndcapsCollection", "EcalEndcapRingCollection",
    "HcalBarrelRegCollection", "HcalEndcapsCollection", "HcalEndcapRingCollection",
    "BeamCalCollection", "LHCalCollection", "LumiCalCollection",
    "YokeBarrelCollection", "YokeEndcapsCollection",
    "ECalBarrelSiHitsEven", "ECalBarrelSiHitsOdd",
    "ECalEndcapSiHitsEven", "ECalEndcapSiHitsOdd",

    "BeamCalCollectionContributions",
    "ECalBarrelScHitsEvenContributions", "ECalBarrelScHitsOddContributions",
    "ECalBarrelSiHitsEvenContributions", "ECalBarrelSiHitsOddContributions",
    "ECalEndcapScHitsEvenContributions", "ECalEndcapScHitsOddContributions",
    "ECalEndcapSiHitsEvenContributions", "ECalEndcapSiHitsOddContributions",
    "EcalEndcapRingCollectionContributions",
    "HcalBarrelRegCollectionContributions", "HcalEndcapsCollectionContributions",
    "HcalEndcapRingCollectionContributions",
    "HCalBarrelRPCHitsContributions", "HCalECRingRPCHitsContributions",
    "HCalEndcapRPCHitsContributions",
    "LHCalCollectionContributions", "LumiCalCollectionContributions",
    "YokeEndcapsCollectionContributions",

    "TPCCollection", "TPCLowPtCollection", "TPCSpacePointCollection",
    "VXDCollection", "SITCollection", "SETCollection", "FTDCollection",
    "FTD_PIXELCollection", "FTD_STRIPCollection",
    "TPCTrackerHitRelations", "VXDTrackerHitRelations", "SITTrackerHitRelations",
    "SETTrackerHitRelations", "FTDPixelTrackerHitRelations",
    "FTDSpacePointRelations", "FTDStripTrackerHitRelations",

    "EcalBarrelRelationsSimDigi", "EcalBarrelRelationsSimRec",
    "EcalEndcapsRelationsSimDigi", "EcalEndcapsRelationsSimRec",
    "EcalEndcapRingRelationsSimDigi", "EcalEndcapRingRelationsSimRec",
    "HcalBarrelRelationsSimDigi", "HcalBarrelRelationsSimRec",
    "HcalEndcapsRelationsSimDigi", "HcalEndcapsRelationsSimRec",
    "HcalEndcapRingRelationsSimDigi", "HcalEndcapRingRelationsSimRec",
    "RelationBCalHit", "RelationLcalHit", "RelationLHcalHit", "RelationMuonHit",
]


def human_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def slim_file(in_path, out_path, max_events):
    infile = ROOT.TFile.Open(in_path)
    tree = infile.Get("events")

    if tree is None:
        raise RuntimeError("Could not find events tree in " + in_path)

    n_total = tree.GetEntries()
    n_keep = n_total if max_events is None else min(max_events, n_total)

    tree.SetBranchStatus("*", 1)

    all_branch_names = [b.GetName() for b in tree.GetListOfBranches()]

    dropped = []
    for pattern in DROP_PATTERNS:
        matches = [b for b in all_branch_names if b.startswith(pattern)]
        if matches:
            tree.SetBranchStatus(pattern + "*", 0)
            dropped.append(pattern)

    outfile = ROOT.TFile.Open(
        out_path, "RECREATE", "",
        ROOT.RCompressionSetting.EAlgorithm.kLZMA
    )
    outfile.SetCompressionLevel(1)

    newtree = tree.CloneTree(n_keep, "fast")
    newtree.Write()
    outfile.Close()
    infile.Close()

    return n_total, n_keep, dropped


def main():
    max_events_label = str(MAX_EVENTS) if MAX_EVENTS else "ALL"

    print("=" * 70)
    print("  slim_reco_files.py")
    print("  MAX_EVENTS = " + max_events_label)
    print("=" * 70)

    for label, paths in FILES.items():
        in_path, out_path = paths["in"], paths["out"]

        if not os.path.exists(in_path):
            print("")
            print("  [" + label.upper() + "] SKIPPED -- input not found: " + in_path)
            continue

        size_before = human_mb(in_path)
        print("")
        print("  [" + label.upper() + "] Processing...")
        print("    Input : " + in_path)
        print("    Size before : %.1f MB" % size_before)

        n_total, n_keep, dropped = slim_file(in_path, out_path, MAX_EVENTS)

        size_after = human_mb(out_path)
        reduction_pct = 100 * (1 - size_after / size_before)

        print("    Events: %d total -> %d kept" % (n_total, n_keep))
        print("    Collections dropped: %d" % len(dropped))
        print("    Output: " + out_path)
        print("    Size after  : %.1f MB" % size_after)
        print("    Reduction   : %.1f%%" % reduction_pct)

    print("")
    print("=" * 70)
    print("  Done. Check output sizes above -- if still over ~50MB,")
    print("  lower MAX_EVENTS further (e.g. 2000-3000) and re-run.")
    print("=" * 70)


if __name__ == "__main__":
    main()
