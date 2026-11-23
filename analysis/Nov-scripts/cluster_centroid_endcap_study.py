#!/usr/bin/env python3
"""Plot photon morphology from the November 6 reduced NPZs.

Run with --compact-dir pointing at the directory containing g4_photons.npz,
cc3_photons.npz, g4_pairs.npz and cc3_pairs.npz. If you have the original
truth/reco/clusters NPZ triplets, first run the November 6 run_study.py.

Uses only NumPy and Matplotlib. Figure uncertainties are single-sample,
event-clustered standard errors on means. See CSVs for denominators and
quantiles. The two samples are summarized independently.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENERGY_EDGES = (10, 20, 30, 40, 60, 80, 100, 125)
SEPARATION_EDGES = (0, 6, 10, 20, np.inf)
REGIONS = ("Full selected acceptance", "Barrel interior", "Endcap interior")
POPULATIONS = ("Quiet photons", "Direct daughters of a two-photon π⁰")
STYLES = {
    "Geant4": ("#2166ac", "o"),
    "CC3 flagged": ("#d6604d", "s"),
    "CC3 unflagged": ("#777777", "x"),
    "CC3 inclusive": ("#d6604d", "s"),
}
PHOTON_KEYS = ("loop", "cid", "idx")
PHOTON_FIELDS = PHOTON_KEYS + (
    "E", "theta", "phi", "electron_parent", "fast", "nearest_photon_mrad",
    "nearest_charged_mrad", "owned_clean", "cluster_idx", "width_ip",
    "width_centered", "centroid_offset",
)
PAIR_FIELDS = (
    "loop", "E", "region", "T", "TE", "outcome", "alpha_true_mrad",
    "alpha_reco_mrad", "A_centroid_out", "B_centroid_out",
    "A_true_cid", "A_true_idx", "B_true_cid", "B_true_idx", "A_fast", "B_fast",
)


def load_rows(directory, sample, kind, fields):
    path = directory / f"{sample}_{kind}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. See the module docstring.")
    with np.load(path, allow_pickle=False) as archive:
        rows = archive["rows"]
    missing = set(fields) - set(rows.dtype.names or ())
    if missing:
        raise ValueError(f"{path} is missing November 6 columns {sorted(missing)}")
    # Copy only used columns so the wide input table can be released.
    result = np.empty(len(rows), dtype=[(key, rows.dtype[key]) for key in fields])
    for key in fields:
        result[key] = rows[key]
    return result


def photon_regions(rows):
    theta = rows["theta"]
    fold = np.minimum(theta, 180 - theta)
    crack = (rows["phi"] + 158.01) % 45 <= 9.01
    return (
        np.isfinite(theta) & np.isfinite(rows["phi"]),
        (theta >= 40) & (theta <= 140) & ~crack,
        (fold >= 15) & (fold < 30),
    )


def photon_populations(photons, pairs):
    # Pair rows include all eligible truth π⁰s, independent of T or TE.
    # Build a packed key array to avoid comparing structured-record padding.
    dtype = np.dtype([(key, photons.dtype[key]) for key in PHOTON_KEYS])
    daughters = np.empty(2 * len(pairs), dtype=dtype)
    keys = np.empty(len(photons), dtype=dtype)
    for key in PHOTON_KEYS:
        keys[key] = photons[key]
    for side, target in (("A", daughters[:len(pairs)]), ("B", daughters[len(pairs):])):
        target["loop"] = pairs["loop"]
        target["cid"] = pairs[f"{side}_true_cid"]
        target["idx"] = pairs[f"{side}_true_idx"]
    quiet = (photons["nearest_photon_mrad"] >= 20) & (photons["nearest_charged_mrad"] >= 20)
    return quiet, np.isin(keys, daughters)


def summarize(values, events):
    if not np.isfinite(values).all():
        raise ValueError("Expected finite response values")
    if not len(values):
        return dict(n_valid=0, n_events=0, mean=np.nan, se_event=np.nan,
                    median=np.nan, p84=np.nan)
    mean = np.mean(values)
    _, event_index = np.unique(events, return_inverse=True)
    residuals = np.bincount(event_index, weights=values - mean)
    n_events = len(residuals)
    se = np.nan
    if n_events > 1:
        se = np.sqrt(n_events / (n_events - 1) * (residuals @ residuals)) / len(values)
    median, p84 = np.quantile(values, (0.5, 0.84))
    return dict(n_valid=len(values), n_events=n_events, mean=mean, se_event=se,
                median=median, p84=p84)


def photon_summary(photons, pairs, sample):
    if sample == "g4":
        flags = {"Geant4": np.ones(len(photons), dtype=bool)}
    else:
        flags = {
            "CC3 inclusive": np.ones(len(photons), dtype=bool),
            "CC3 flagged": photons["fast"] == 1,
            "CC3 unflagged": photons["fast"] == 0,
            "CC3 unknown flag": ~np.isin(photons["fast"], (0, 1)),
        }
    rows = []
    base = photons["electron_parent"] == 0
    memberships = photon_populations(photons, pairs)
    for region, geometry in zip(REGIONS, photon_regions(photons)):
        for pop, membership in zip(POPULATIONS, memberships):
            population = base & geometry & membership
            for lo, hi in zip(ENERGY_EDGES, ENERGY_EDGES[1:]):
                energy_bin = population & (photons["E"] >= lo) & (photons["E"] < hi)
                for label, flag in flags.items():
                    truth = energy_bin & flag
                    clean = truth & (photons["owned_clean"] == 1)
                    has_cluster = clean & (photons["cluster_idx"] >= 0)
                    for field in ("width_ip", "centroid_offset", "width_centered"):
                        valid = has_cluster & np.isfinite(photons[field])
                        rows.append(dict(
                            region=region, population=pop, energy_lo=lo, energy_hi=hi,
                            sample=label, quantity=field, n_truth=int(truth.sum()),
                            n_clean=int(clean.sum()), n_first_cluster=int(has_cluster.sum()),
                            **summarize(photons[field][valid], photons["loop"][valid]),
                        ))
    return rows


def pair_summary(pairs, sample):
    if sample == "g4":
        flags = {"Geant4": np.ones(len(pairs), dtype=bool)}
    else:
        known = np.isin(pairs["A_fast"], (0, 1)) & np.isin(pairs["B_fast"], (0, 1))
        count = pairs["A_fast"] + pairs["B_fast"]
        flags = {
            "CC3 inclusive": np.ones(len(pairs), dtype=bool),
            "CC3 both flagged": known & (count == 2),
            "CC3 higher-energy only": known & (pairs["A_fast"] == 1) & (count == 1),
            "CC3 lower-energy only": known & (pairs["B_fast"] == 1) & (count == 1),
            "CC3 neither flagged": known & (count == 0),
            "CC3 unknown flag": ~known,
        }
    # Focused replication of the existing endcap mechanism slice.
    base = np.isin(pairs["region"], (2, 3)) & (pairs["E"] >= 30) & (pairs["E"] < 50)
    opening = pairs["alpha_reco_mrad"] - pairs["alpha_true_mrad"]
    values = {"opening_error": opening, "higher_centroid_out": pairs["A_centroid_out"],
              "lower_centroid_out": pairs["B_centroid_out"]}
    rows = []
    for lo, hi in zip(SEPARATION_EDGES, SEPARATION_EDGES[1:]):
        separation = (pairs["alpha_true_mrad"] >= lo) & (pairs["alpha_true_mrad"] < hi)
        for label, flag in flags.items():
            truth = base & separation & flag
            resolved = truth & (pairs["T"] == 1)
            eligible = truth & (pairs["TE"] == 1)
            for quantity, value in values.items():
                valid = eligible & np.isfinite(value)
                rows.append(dict(
                    separation_lo=lo, separation_hi=hi, sample=label, quantity=quantity,
                    n_truth=int(truth.sum()), n_resolved=int(resolved.sum()),
                    n_energy_eligible=int(eligible.sum()),
                    n_shared_best_pfo=int((truth & (pairs["outcome"] == 2)).sum()),
                    **summarize(value[valid], pairs["loop"][valid]),
                ))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_photons(rows, output, quantity, title, ylabel, min_valid):
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True, layout="constrained")
    labels = ("Geant4", "CC3 flagged", "CC3 unflagged")
    centers = np.sqrt(np.array(ENERGY_EDGES[:-1]) * ENERGY_EDGES[1:])
    for region, panels in zip(REGIONS, axes):
        for pop, ax in zip(POPULATIONS, panels):
            for label in labels:
                selected = [row for row in rows if row["region"] == region
                            and row["population"] == pop and row["sample"] == label
                            and row["quantity"] == quantity]
                color, marker = STYLES[label]
                means = [r["mean"] if r["n_valid"] >= min_valid else np.nan for r in selected]
                errors = [r["se_event"] if r["n_valid"] >= min_valid else np.nan for r in selected]
                ax.errorbar(centers, means, yerr=errors, color=color, marker=marker,
                            markersize=4, linewidth=1.2, capsize=2, label=label)
            ax.set_title(f"{region} · {pop}", fontsize=9)
            ax.set_xscale("log")
            ax.set_xticks((10, 20, 40, 80, 125), labels=("10", "20", "40", "80", "125"))
            ax.grid(alpha=0.2)
        panels[0].set_ylabel(ylabel)
    for ax in axes[-1]:
        ax.set_xlabel("True photon energy [GeV]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.supxlabel(
        f"Clean photon PFOs, first cluster · Bars show event-level SE · Bins shown with ≥{min_valid} valid photons",
        fontsize=9,
    )
    fig.savefig(output / f"{quantity}.png", dpi=160)
    plt.close(fig)


def plot_pairs(rows, output, min_valid):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), sharex=True, layout="constrained")
    categories = ("0–6", "6–10", "10–20", "≥20")
    for ax, (quantity, title) in zip(axes, (
        ("higher_centroid_out", "Higher-energy daughter"),
        ("lower_centroid_out", "Lower-energy daughter"),
        ("opening_error", "Reconstructed opening-angle error"),
    )):
        for label in ("Geant4", "CC3 inclusive"):
            selected = [r for r in rows if r["quantity"] == quantity and r["sample"] == label]
            means = [r["mean"] if r["n_valid"] >= min_valid else np.nan for r in selected]
            errors = [r["se_event"] if r["n_valid"] >= min_valid else np.nan for r in selected]
            color, marker = STYLES[label]
            ax.errorbar(categories, means, yerr=errors, color=color, marker=marker,
                        linewidth=1.2, capsize=2, label=label)
        ax.axhline(0, color="0.55", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("True γγ separation [mrad]")
    axes[0].set_ylabel("Signed error [mrad]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Endcap π⁰ pairs, 30–50 GeV true π⁰ energy")
    fig.supxlabel(
        f"Resolved pairs before the mass window, each photon ≥0.5 GeV · Event-level SE · ≥{min_valid} valid pairs per point",
        fontsize=9,
    )
    fig.savefig(output / "endcap_pair_direction.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--min-valid", type=int, default=20, help="Plot threshold; CSVs keep all bins")
    args = parser.parse_args()
    if args.min_valid < 1:
        parser.error("--min-valid must be positive")
    photon_rows = []
    pair_rows = []
    for sample in ("g4", "cc3"):
        photons = load_rows(args.compact_dir, sample, "photons", PHOTON_FIELDS)
        pairs = load_rows(args.compact_dir, sample, "pairs", PAIR_FIELDS)
        photon_rows.extend(photon_summary(photons, pairs, sample))
        pair_rows.extend(pair_summary(pairs, sample))
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "photon_response.csv", photon_rows)
    write_csv(args.outdir / "endcap_pair_direction.csv", pair_rows)
    plot_photons(photon_rows, args.outdir, "width_ip",
                 "How wide are reconstructed photon clusters?", "Mean transverse RMS [mm]", args.min_valid)
    plot_photons(photon_rows, args.outdir, "centroid_offset",
                 "How far are cluster centres from the true photon trajectory?",
                 "Mean centroid offset [mm]", args.min_valid)
    plot_pairs(pair_rows, args.outdir, args.min_valid)
    print(f"Saved 3 figures and 2 tables in {args.outdir}")


if __name__ == "__main__":
    main()
