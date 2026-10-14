"""
extract_to_parquet.py -- Direct ROOT -> Parquet extraction using uproot/awkward.
Uses substring-based branch discovery (robust to exact naming quirks)
instead of guessing exact branch strings.

Run on NAF:
    python3 extract_to_parquet.py
"""

import uproot
import awkward as ak
import os

MAX_EVENTS = 100  # start small to sanity check, then bump up

FILES = {
    "g4":  "/data/dust/user/alimuham/thesis/reco/tau-pi0-g4-taureco-10k-events-test_REC.edm4hep.root",
    "cc3": "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root",
}

TREE_NAME = "events"

# (output_field_name, collection_prefix, field_substring)
# field_substring=None means "just the bare collection branch" (rare)
WANTED_FIELDS = [
    ("ggpi0_energy", "GammaGammaCandidatePi0s", "energy"),
    ("ggpi0_mass",   "GammaGammaCandidatePi0s", "mass"),
    ("ggpi0_px",     "GammaGammaCandidatePi0s", "momentum.x"),
    ("ggpi0_py",     "GammaGammaCandidatePi0s", "momentum.y"),
    ("ggpi0_pz",     "GammaGammaCandidatePi0s", "momentum.z"),

    ("pfo_pdg",    "PandoraPFOs", "PDG"),
    ("pfo_energy", "PandoraPFOs", "energy"),
    ("pfo_px",     "PandoraPFOs", "momentum.x"),
    ("pfo_py",     "PandoraPFOs", "momentum.y"),
    ("pfo_pz",     "PandoraPFOs", "momentum.z"),
    ("pfo_mass",   "PandoraPFOs", "mass"),

    ("mc_pdg",       "MCParticles", "PDG"),
    ("mc_genstatus", "MCParticles", "generatorStatus"),
    ("mc_mass",      "MCParticles", "mass"),
    ("mc_px",        "MCParticles", "momentum.x"),
    ("mc_py",        "MCParticles", "momentum.y"),
    ("mc_pz",        "MCParticles", "momentum.z"),
    ("mc_parents_begin", "MCParticles", "parentsbegin"),
    ("mc_parents_end",   "MCParticles", "parentsend"),

    ("link_weight", "RecoMCTruthLink", "weight"),
]

# Relation index branches (from/to object references) -- searched separately
# since their naming pattern differs (collection + relation name, no dot field)
RELATION_FIELDS = [
    ("mc_parent_idx",  "MCParticles", "parents", "index"),
    ("link_from_idx",  "RecoMCTruthLink", "from", "index"),
    ("link_to_idx",    "RecoMCTruthLink", "to", "index"),
]


def list_all_branches(tree):
    """Get every branch/leaf name uproot knows about, recursively."""
    names = set()
    for k in tree.keys(recursive=True):
        names.add(k)
    return names


def find_best_match(all_names, collection, field):
    """Find a branch name containing both collection and field substrings,
    preferring the shortest match (least nested / most direct)."""
    candidates = [n for n in all_names if collection in n and field in n]
    if not candidates:
        return None
    candidates.sort(key=len)
    return candidates[0]


def extract_file(path, label, max_events):
    print(f"\n[{label.upper()}] Opening: {path}")
    f = uproot.open(path)
    tree = f[TREE_NAME]
    n_total = tree.num_entries
    n_read = n_total if max_events is None else min(max_events, n_total)
    print(f"  Total events: {n_total}, reading: {n_read}")

    all_names = list_all_branches(tree)
    print(f"  Total branches/leaves discovered: {len(all_names)}")

    out = {}
    resolved = {}
    for out_key, collection, field in WANTED_FIELDS:
        match = find_best_match(all_names, collection, field)
        if match is None:
            print(f"  MISS: {out_key} (collection={collection}, field={field})")
            out[out_key] = None
            continue
        try:
            arr = tree[match].array(entry_stop=n_read)
            out[out_key] = arr
            resolved[out_key] = match
        except Exception as e:
            print(f"  ERROR reading {match} for {out_key}: {e}")
            out[out_key] = None

    for out_key, collection, relname, subfield in RELATION_FIELDS:
        match = find_best_match(all_names, collection, relname)
        if match is None:
            print(f"  MISS relation: {out_key} (collection={collection}, rel={relname})")
            out[out_key] = None
            continue
        # try to find the .index sub-leaf specifically
        sub_match = find_best_match(all_names, match, subfield)
        target = sub_match if sub_match else match
        try:
            arr = tree[target].array(entry_stop=n_read)
            out[out_key] = arr
            resolved[out_key] = target
        except Exception as e:
            print(f"  ERROR reading {target} for {out_key}: {e}")
            out[out_key] = None

    found = [k for k, v in out.items() if v is not None]
    missing = [k for k, v in out.items() if v is None]
    print(f"  Resolved fields: {len(found)} / {len(out)}")
    if resolved:
        print("  Branch name mapping used:")
        for k, v in resolved.items():
            print(f"    {k:20s} -> {v}")
    if missing:
        print(f"  Still missing: {missing}")

    return out, n_read, sorted(all_names)


def build_record_array(fields):
    clean = {k: v for k, v in fields.items() if v is not None}
    if not clean:
        raise RuntimeError("Nothing extracted -- see branch dump file for available names.")
    return ak.zip(clean, depth_limit=1)


def save_parquet(arr, out_path):
    ak.to_parquet(arr, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved: {out_path}  ({size_mb:.3f} MB)")


def main():
    print("=" * 70)
    print("  extract_to_parquet.py (substring-matching version)")
    print(f"  MAX_EVENTS = {MAX_EVENTS if MAX_EVENTS is not None else 'ALL'}")
    print("=" * 70)

    results = {}
    for label, path in FILES.items():
        if not os.path.exists(path):
            print(f"\n[{label.upper()}] SKIPPED -- not found: {path}")
            continue
        fields, n_read, all_names = extract_file(path, label, MAX_EVENTS)

        # dump full branch list once, for manual inspection if needed
        dump_path = f"{label}_all_branches.txt"
        with open(dump_path, "w") as fh:
            fh.write("\n".join(all_names))
        print(f"  Full branch list written to: {dump_path}")

        arr = build_record_array(fields)
        out_path = f"{label}_pi0_extract.parquet"
        save_parquet(arr, out_path)
        results[label] = arr

    if len(results) == 2:
        g4_arr = ak.with_field(results["g4"], "g4", where="sample")
        cc3_arr = ak.with_field(results["cc3"], "cc3", where="sample")
        combined = ak.concatenate([g4_arr, cc3_arr], axis=0)
        save_parquet(combined, "pi0_extract_combined.parquet")

    print("\n" + "=" * 70)
    print("  Done. If any fields are still MISSING above, open the")
    print("  *_all_branches.txt files and grep for the collection name")
    print("  to find the exact branch string, then tell me what you see.")
    print("=" * 70)


if __name__ == "__main__":
    main()
