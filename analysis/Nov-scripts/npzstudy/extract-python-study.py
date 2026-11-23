#!/usr/bin/env python3
"""
made by chatgpt to extract root data into npz files for faster plotting an analysis 

Extract linked numerical tables from EDM4hep REC files.

Defaults:
  Both baseline G4 and CC3 files.
  Output under reco/npz-data-summary/<tag>/.
  Three compressed NPZ files per event chunk.

Units:
  Energy/momentum: GeV
  Positions: mm
  Time: native EDM4hep units
  Four-momenta and positions: float64

Join keys:
  Within one input file:
    event_loop, collectionID, object index
  Never join G4 and CC3 using object indices.

No physics selections or truth-link thresholds are applied.
No Python object arrays are written.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


REC_DIR = Path("/data/dust/user/alimuham/thesis/reco")

DEFAULT_FILES = {
    "g4": REC_DIR / "tau-pi0-geant4-100kevents-reco-test_REC.edm4hep.root",
    "cc3": REC_DIR / "tau-pi0-caloclouds-100kevents-reco-test_REC.edm4hep.root",
}

# field:type. All IDs are signed int64 so uint32 collection IDs remain intact.
SCHEMAS = {
    "events": "loop:i run:i event:i n_mc:i n_pfo:i n_candidate:i",
    "mc": (
        "loop:i cid:i idx:i pdg:i gen:i sim_status:i fast:i "
        "E:f px:f py:f pz:f mass:f charge:f "
        "vx:f vy:f vz:f ex:f ey:f ez:f time:f"
    ),
    "ancestry": "loop:i child_cid:i child_idx:i parent_cid:i parent_idx:i",
    "pfo": (
        "loop:i cid:i idx:i pdg:i E:f px:f py:f pz:f "
        "mass:f charge:f goodness:f"
    ),
    "links": (
        "loop:i pfo_cid:i pfo_idx:i mc_cid:i mc_idx:i "
        "raw:f cw:f tw:f"
    ),
    "pfo_cluster": (
        "loop:i pfo_cid:i pfo_idx:i order:i cluster_cid:i cluster_idx:i"
    ),
    "candidate": (
        "loop:i cid:i idx:i pdg:i E:f px:f py:f pz:f "
        "mass:f goodness:f n_daughters:i"
    ),
    "candidate_daughter": (
        "loop:i candidate_cid:i candidate_idx:i order:i "
        "pfo_cid:i pfo_idx:i"
    ),
    "cluster": (
        "loop:i cid:i idx:i E:f x:f y:f z:f n_hits:i "
        "hit_E_sum:f mean_x:f mean_y:f mean_z:f "
        "mean_xx:f mean_yy:f mean_zz:f mean_xy:f mean_xz:f mean_yz:f"
    ),
    "hit": (
        "loop:i cid:i idx:i E:f x:f y:f z:f time:f type:i"
    ),
    "cluster_hit": (
        "loop:i cluster_cid:i cluster_idx:i order:i hit_cid:i hit_idx:i"
    ),
}

GROUPS = {
    "truth": ["events", "mc", "ancestry"],
    "reco": [
        "events", "pfo", "links", "pfo_cluster",
        "candidate", "candidate_daughter",
    ],
    "clusters": ["events", "cluster", "hit", "cluster_hit"],
}


class Table:
    def __init__(self, schema):
        self.fields = [item.split(":") for item in schema.split()]
        self.columns = [[] for _ in self.fields]

    def add(self, *values):
        if len(values) != len(self.fields):
            raise ValueError("Row does not match schema")
        for column, value in zip(self.columns, values):
            column.append(value)

    def arrays(self, prefix):
        return {
            f"{prefix}_{name}": np.asarray(
                values, dtype=np.int64 if kind == "i" else np.float64
            )
            for (name, kind), values in zip(self.fields, self.columns)
        }

    def __len__(self):
        return len(self.columns[0])


def new_tables():
    return {name: Table(schema) for name, schema in SCHEMAS.items()}


def object_id(obj):
    if hasattr(obj, "isAvailable") and not obj.isAvailable():
        raise RuntimeError("Encountered an unavailable referenced object")
    oid = obj.getObjectID()
    key = int(oid.collectionID), int(oid.index)
    if key[1] < 0:
        raise RuntimeError(f"Invalid object ID {key}")
    return key


def xyz(vector):
    return float(vector.x), float(vector.y), float(vector.z)


def pdg(particle):
    if hasattr(particle, "getPDG"):
        return int(particle.getPDG())
    return int(particle.getType())


def optional_scalar(obj, method, missing):
    function = getattr(obj, method, None)
    return missing if function is None else function()


def four_vector(particle):
    px, py, pz = xyz(particle.getMomentum())
    return float(particle.getEnergy()), px, py, pz


def link_ends(link):
    if hasattr(link, "getRec"):
        return link.getRec(), link.getSim()
    return link.getFrom(), link.getTo()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def atomic_npz(path, arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_cluster(cluster, loop, tables, seen_clusters, seen_hits):
    cid, idx = object_id(cluster)
    if (cid, idx) in seen_clusters:
        return
    seen_clusters.add((cid, idx))

    positions, energies = [], []
    hits = list(cluster.getHits())
    for order, hit in enumerate(hits):
        hit_cid, hit_idx = object_id(hit)
        tables["cluster_hit"].add(loop, cid, idx, order, hit_cid, hit_idx)

        position = xyz(hit.getPosition())
        energy = float(hit.getEnergy())
        positions.append(position)
        energies.append(energy)

        if (hit_cid, hit_idx) not in seen_hits:
            seen_hits.add((hit_cid, hit_idx))
            tables["hit"].add(
                loop, hit_cid, hit_idx, energy, *position,
                float(optional_scalar(hit, "getTime", np.nan)),
                int(optional_scalar(hit, "getType", -1)),
            )

    # Raw energy-weighted moments reproduce the old centroid-axis RMS.
    # Raw hits also permit other axes, profiles and pair-facing asymmetries.
    total = float(np.sum(energies))
    moments = [np.nan] * 9
    if hits and total > 0 and np.isfinite(total):
        position = np.asarray(positions, dtype=np.float64)
        weights = np.asarray(energies, dtype=np.float64) / total
        mean = weights @ position
        second = (position * weights[:, None]).T @ position
        moments = [
            *mean,
            second[0, 0], second[1, 1], second[2, 2],
            second[0, 1], second[0, 2], second[1, 2],
        ]

    tables["cluster"].add(
        loop, cid, idx, float(cluster.getEnergy()),
        *xyz(cluster.getPosition()), len(hits), total, *moments,
    )


def extract_event(frame, loop, tables, counters, args):
    # Required collections fail explicitly if absent.
    mcs = frame.get(args.mc_collection)
    pfos = frame.get(args.pfo_collection)
    links = frame.get(args.link_collection)
    candidates = frame.get(args.candidate_collection)
    header = frame.get("EventHeader")

    if len(header) != 1:
        raise RuntimeError(f"Event {loop}: expected one EventHeader")

    tables["events"].add(
        loop, int(header[0].getRunNumber()), int(header[0].getEventNumber()),
        len(mcs), len(pfos), len(candidates),
    )

    mc_ids = {object_id(mc) for mc in mcs}
    pfo_ids = {object_id(pfo) for pfo in pfos}
    if len(mc_ids) != len(mcs) or len(pfo_ids) != len(pfos):
        raise RuntimeError(f"Event {loop}: duplicate object IDs")

    for mc in mcs:
        cid, idx = object_id(mc)
        fast = int(optional_scalar(mc, "isHandledByFastSim", -1))
        if fast not in (-1, 0, 1):
            raise RuntimeError(f"Unexpected fast-simulation flag {fast}")
        counters["mc_fast_unknown"] += int(fast == -1)

        tables["mc"].add(
            loop, cid, idx, int(mc.getPDG()), int(mc.getGeneratorStatus()),
            int(optional_scalar(mc, "getSimulatorStatus", -1)), fast,
            *four_vector(mc), float(mc.getMass()), float(mc.getCharge()),
            *xyz(mc.getVertex()), *xyz(mc.getEndpoint()),
            float(optional_scalar(mc, "getTime", np.nan)),
        )
        # Preserve raw endpoints, including (0, 0, 0).
        for parent in mc.getParents():
            parent_id = object_id(parent)
            if parent_id not in mc_ids:
                raise RuntimeError(f"Event {loop}: parent outside MC collection")
            tables["ancestry"].add(loop, cid, idx, *parent_id)

        if mc.getPDG() == 22 and mc.getGeneratorStatus() == 1:
            counters["generator_photons"] += 1
            if mc.getEnergy() >= 10:
                counters["generator_photons_Ege10"] += 1
                electron_parent = any(
                    abs(parent.getPDG()) == 11 for parent in mc.getParents()
                )
                if not electron_parent:
                    counters["generator_photons_Ege10_no_e_parent"] += 1

    seen_clusters, seen_hits = set(), set()
    for pfo in pfos:
        cid, idx = object_id(pfo)
        tables["pfo"].add(
            loop, cid, idx, pdg(pfo), *four_vector(pfo),
            float(pfo.getMass()), float(pfo.getCharge()),
            float(optional_scalar(pfo, "getGoodnessOfPID", np.nan)),
        )
        for order, cluster in enumerate(pfo.getClusters()):
            tables["pfo_cluster"].add(
                loop, cid, idx, order, *object_id(cluster)
            )
            extract_cluster(
                cluster, loop, tables, seen_clusters, seen_hits
            )

    # Includes Pandora clusters not referenced by a PFO.
    for cluster in frame.get(args.cluster_collection):
        extract_cluster(cluster, loop, tables, seen_clusters, seen_hits)

    for link in links:
        reco, truth = link_ends(link)
        reco_id, truth_id = object_id(reco), object_id(truth)
        if reco_id not in pfo_ids or truth_id not in mc_ids:
            raise RuntimeError(
                f"Event {loop}: truth link refers outside configured collections"
            )
        raw = float(link.getWeight())
        if not np.isfinite(raw) or raw < 0:
            raise RuntimeError(f"Event {loop}: invalid truth-link weight {raw}")
        packed = int(round(raw))
        cw, tw = (packed // 10000) / 1000.0, (packed % 10000) / 1000.0
        counters["link_weight_above_one"] += int(cw > 1.001 or tw > 1.001)
        tables["links"].add(loop, *reco_id, *truth_id, raw, cw, tw)

    for candidate in candidates:
        cid, idx = object_id(candidate)
        daughters = list(candidate.getParticles())
        goodness = float(
            optional_scalar(candidate, "getGoodnessOfPID", np.nan)
        )
        counters["candidate_goodness_missing"] += int(not np.isfinite(goodness))
        tables["candidate"].add(
            loop, cid, idx, pdg(candidate), *four_vector(candidate),
            float(candidate.getMass()), goodness, len(daughters),
        )
        for order, daughter in enumerate(daughters):
            daughter_id = object_id(daughter)
            if daughter_id not in pfo_ids:
                raise RuntimeError(
                    f"Event {loop}: candidate daughter outside PFO collection"
                )
            tables["candidate_daughter"].add(
                loop, cid, idx, order, *daughter_id
            )


def flush(tables, directory, tag, part, manifest):
    """Write all three parts, then mark the chunk complete in the manifest."""
    records = []
    for group, names in GROUPS.items():
        arrays = {}
        for name in names:
            arrays.update(tables[name].arrays(name))
        arrays["meta_schema_version"] = np.asarray(1, dtype=np.int64)
        arrays["meta_sample"] = np.asarray(tag)

        filename = f"{tag}_{group}_{part:05d}.npz"
        path = directory / filename
        atomic_npz(path, arrays)
        records.append({
            "file": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": {name: len(tables[name]) for name in names},
        })

    loops = tables["events"].columns[0]
    manifest["chunks"].append({
        "part": part,
        "first_event_loop": loops[0],
        "last_event_loop": loops[-1],
        "files": records,
    })
    atomic_json(directory / "manifest.json", manifest)
    total_mb = sum(record["bytes"] for record in records) / 1024**2
    print(
        f"{tag}: saved events {loops[0]}–{loops[-1]}, "
        f"three files, {total_mb:.1f} MiB",
        flush=True,
    )


def run(tag, source, args):
    from podio import root_io

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    directory = args.outdir / tag
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {directory}\n"
            "Use a different --outdir to avoid mixing runs."
        )
    directory.mkdir(parents=True, exist_ok=True)

    stat = source.stat()
    manifest = {
        "schema_version": 1,
        "sample": tag,
        "source": str(source),
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "script_sha256": sha256(Path(__file__)),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "schemas": SCHEMAS,
        "groups": GROUPS,
        "units": {"energy": "GeV", "momentum": "GeV", "position": "mm"},
        "notes": [
            "No physics selection applied.",
            "All MCParticles and all PandoraPFOs retained.",
            "Object IDs valid only within input file and event_loop.",
            "fast=-1 means API unavailable; never interpreted as fallback.",
            "Raw endpoints preserved; zero endpoints require interpretation.",
            "goodness is raw stored goodnessOfPID; meaning requires provenance.",
            "Cluster moments use raw hit energy weights.",
            "Only reconstructed cluster hits are stored, not simulated hits.",
            "Only chunks listed here are complete three-file sets.",
        ],
        "chunks": [],
        "complete": False,
        "counters": {},
    }
    atomic_json(directory / "manifest.json", manifest)

    reader_class = getattr(root_io, "Reader", None)
    if reader_class is None:
        reader_class = root_io.RootReader
    reader = reader_class(str(source))

    tables, counters = new_tables(), Counter()
    part = 0
    try:
        for loop, frame in enumerate(reader.get("events")):
            if args.max_events and loop >= args.max_events:
                break
            extract_event(frame, loop, tables, counters, args)
            counters["events_processed"] += 1

            if len(tables["events"]) >= args.chunk_events:
                manifest["counters"] = dict(counters)
                flush(tables, directory, tag, part, manifest)
                tables = new_tables()
                part += 1

        if len(tables["events"]):
            manifest["counters"] = dict(counters)
            flush(tables, directory, tag, part, manifest)

        manifest["complete"] = True
        manifest["counters"] = dict(counters)
        atomic_json(directory / "manifest.json", manifest)

    except BaseException as error:
        # A partially extracted current chunk is intentionally not committed.
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["counters_at_failure"] = dict(counters)
        atomic_json(directory / "manifest.json", manifest)
        raise

    print(f"{tag}: finished. Output: {directory}", flush=True)
    print(json.dumps(dict(counters), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", choices=["g4", "cc3", "both"], default="both")
    parser.add_argument("--g4", type=Path, default=DEFAULT_FILES["g4"])
    parser.add_argument("--cc3", type=Path, default=DEFAULT_FILES["cc3"])
    parser.add_argument("--outdir", type=Path, default=REC_DIR / "npz-data-summary")
    parser.add_argument("--max-events", type=int, default=0, help="0 means all")
    parser.add_argument("--chunk-events", type=int, default=1000)
    parser.add_argument("--mc-collection", default="MCParticles")
    parser.add_argument("--pfo-collection", default="PandoraPFOs")
    parser.add_argument("--link-collection", default="RecoMCTruthLink")
    parser.add_argument("--cluster-collection", default="PandoraClusters")
    parser.add_argument("--candidate-collection", default="GammaGammaCandidatePi0s")
    args = parser.parse_args()

    if args.max_events < 0 or args.chunk_events <= 0:
        parser.error("max-events must be >=0 and chunk-events must be >0")

    for tag in (["g4", "cc3"] if args.which == "both" else [args.which]):
        run(tag, getattr(args, tag), args)


if __name__ == "__main__":
    main()
