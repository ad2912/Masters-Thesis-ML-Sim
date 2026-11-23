#!/usr/bin/env python3
"""uds GEN-level first look: 12 AllShowers species, three energy cuts, two regions.

Read converted EDM4hep ROOT files on DUST. Save CSV and PNG summaries on AFS.
This counts GEN-final (generator status 1) particles; it does NOT establish
which particles the deployed AllShowers simulation will actually handle.
All input events are pooled WITHOUT cross-section or polarisation weights.
"""

from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from math import atan2, degrees, hypot, isfinite, ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from podio.reading import get_reader


# === SETTINGS: edit here, then run python3 uds_gen_study_v2.py ===
INPUT_DIR = Path("/data/dust/user/alimuham/thesis/InputFiles/uds-study")
OUTPUT_DIR = Path("/afs/desy.de/user/a/alimuham/thesis-ml-sim/uds-plots")
MAX_EVENTS_PER_FILE = None      # None -> all events in all 24 files
EXPECTED_FILES = 24

# Only affects figures (not data reading or CSV summaries).
# For a quick look use ("E10",); for every figure keep all three.
DETAIL_CUTS = ("no_cut", "E5", "E10") 

SPECIES = {
    11: "e-", -11: "e+", 22: "gamma",
    211: "pi+", -211: "pi-",
    321: "K+", -321: "K-", 130: "K0L",
    2212: "p", -2212: "pbar",
    2112: "n", -2112: "nbar",
}
ENERGY_CUTS = {"no_cut": 0.0, "E5": 5.0, "E10": 10.0}
ENERGY_LABELS = {"no_cut": "no energy cut", "E5": "E >= 5 GeV", "E10": "E >= 10 GeV"}
REGIONS = ( "all_angles","barrel") #

# Histograms have a common binning for comparing the panels.
BINS = {
    "energy": np.linspace(0, 260, 131),   # 2 GeV bins
    "theta": np.linspace(0, 180, 61),     # 3 degree bins
    "phi": np.linspace(-180, 180, 73),   # 5 degree bins
}
# Python lists make bisect fast; we find each particle's bins only once.
BIN_EDGES_LIST = {name: edges.tolist() for name, edges in BINS.items()}

SELECTIONS = [(region, cut, pdg)
              for region in REGIONS for cut in ENERGY_CUTS for pdg in SPECIES]

# Accumulators; independent of the number of input events.
multiplicities = {key: Counter() for key in SELECTIONS}
kinematics = {
    (region, cut, pdg, variable): np.zeros(len(edges) - 1, dtype=np.int64)
    for region, cut, pdg in SELECTIONS for variable, edges in BINS.items()
}
particle_totals = Counter()
sum_energies = defaultdict(float)
events_with_any = Counter()  # At least one of the 12 species in an event
out_of_plot_range = Counter()
n_events = 0
n_final = 0
n_supported = 0
invalid_energy = 0
undefined_direction = 0


def in_barrel(theta, phi):
    """User's GEN momentum acceptance (degrees), NOT calorimeter entry."""
    if not (40.0 <= theta <= 140.0):
        return False
    for k in range(8):
        center = -157.5 + 45.0 * k
        if center - 0.51 <= phi <= center + 8.5:
            return False
    return True


def get_angles(particle):
    p = particle.getMomentum()
    px, py, pz = float(p.x), float(p.y), float(p.z)
    norm = hypot(px, py, pz)
    if not isfinite(norm) or norm == 0:
        return None
    theta = degrees(atan2(hypot(px, py), pz))
    phi = degrees(atan2(py, px))
    return theta, -180.0 if phi == 180.0 else phi


def bin_index(variable, value):
    """Return the histogram bin number, or None outside the plotted range."""
    edges = BIN_EDGES_LIST[variable]
    i = bisect_right(edges, value) - 1
    if value == edges[-1]:
        i -= 1  # Include the right-hand endpoint of the final bin.
    return i if 0 <= i < len(edges) - 1 else None


def analyse_event(particles):
    global n_final, n_supported, invalid_energy, undefined_direction
    event_counts = Counter()

    for particle in particles:
        if particle.getGeneratorStatus() != 1:
            continue
        n_final += 1
        pdg = particle.getPDG()
        if pdg not in SPECIES:
            continue
        n_supported += 1

        energy = float(particle.getEnergy())
        if not isfinite(energy) or energy < 0:
            invalid_energy += 1
            continue

        angles = get_angles(particle)
        if angles is None:
            undefined_direction += 1
            regions = ("all_angles",)
            bin_numbers = {"energy": bin_index("energy", energy)}
        else:
            theta, phi = angles
            regions = ("all_angles", "barrel") if in_barrel(theta, phi) else ("all_angles",)
            # Compute once per particle; reuse across up to 6 selections.
            bin_numbers = {
                "energy": bin_index("energy", energy),
                "theta": bin_index("theta", theta),
                "phi": bin_index("phi", phi),
            }

        for region in regions:
            for cut, threshold in ENERGY_CUTS.items():
                if energy < threshold:
                    continue
                key = (region, cut, pdg)
                event_counts[key] += 1
                particle_totals[key] += 1
                sum_energies[key] += energy
                for variable, i in bin_numbers.items():
                    if i is not None:
                        kinematics[(region, cut, pdg, variable)][i] += 1
                    else:
                        out_of_plot_range[variable] += 1

    # Store per-event multiplicities (including zeros later), and event coverage.
    for key, count in event_counts.items():
        multiplicities[key][count] += 1
    for region_cut in {(region, cut) for region, cut, _ in event_counts}:
        events_with_any[region_cut] += 1


def multiplicity_hist(key):
    hist = multiplicities[key].copy()
    hist[0] = n_events - sum(hist.values())
    return hist


def mean_text(value):
    if value < 0.01:
        return f"{value:.4f}"
    if value < 0.1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def annotate_panel(ax, region, cut, pdg, n_plotted=None):
    key = (region, cut, pdg)
    total = particle_totals[key]
    mean = total / n_events
    text = f"N = {total:,}\nmean = {mean_text(mean)}/evt"
    if n_plotted is not None and n_plotted != total:
        text += f"\nplotted = {n_plotted:,}"
    ax.text(0.97, 0.97, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})


def save_csvs(folder, processed_files):
    with (folder / "input_files.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["file", "events_processed", "seconds_total", "seconds_reading"])
        writer.writerows(processed_files)

    with (folder / "species_summary.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["region", "energy_cut", "pdg", "species", "events",
                         "total_particles", "mean_per_event", "events_with_ge1",
                         "fraction_events_with_ge1", "mean_energy_GeV",
                         "barrel_fraction_of_all_angles"])
        for region, cut, pdg in SELECTIONS:
            key = (region, cut, pdg)
            count = particle_totals[key]
            hist = multiplicity_hist(key)
            nonzero = n_events - hist[0]
            baseline = particle_totals[("all_angles", cut, pdg)]
            fraction = (particle_totals[("barrel", cut, pdg)] / baseline
                        if baseline else "")
            writer.writerow([region, cut, pdg, SPECIES[pdg], n_events,
                             count, count / n_events, nonzero, nonzero / n_events,
                             sum_energies[key] / count if count else "", fraction])

    with (folder / "multiplicity_distribution.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["region", "energy_cut", "pdg", "species", "N_per_event",
                         "events", "fraction_of_all_events"])
        for region, cut, pdg in SELECTIONS:
            for count, events in sorted(multiplicity_hist((region, cut, pdg)).items()):
                writer.writerow([region, cut, pdg, SPECIES[pdg], count,
                                 events, events / n_events])

    with (folder / "selection_summary.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["region", "energy_cut", "events", "total_particles",
                         "mean_particles_per_event", "events_with_any_species",
                         "fraction_events_with_any_species"])
        for cut in ENERGY_CUTS:
            for region in REGIONS:
                total = sum(particle_totals[(region, cut, pdg)] for pdg in SPECIES)
                count_events = events_with_any[(region, cut)]
                writer.writerow([region, cut, n_events, total, total / n_events,
                                 count_events, count_events / n_events])


def save_overview(folder):
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.5))
    pdgs = list(SPECIES)
    xs = np.arange(len(pdgs))
    for ax, cut in zip(axes, ENERGY_CUTS):
        for offset, region, label in [(-0.18, "all_angles", "All angles"),
                                      (0.18, "barrel", "Barrel")]:
            means = [particle_totals[(region, cut, pdg)] / n_events for pdg in pdgs]
            bars = ax.bar(xs + offset, means, width=0.36, label=label)
            ax.bar_label(bars, labels=[mean_text(v) for v in means],
                         fontsize=7, padding=2)
        ax.set_xticks(xs, [SPECIES[pdg] for pdg in pdgs])
        ax.set_ylabel("Mean particles / event")
        ax.set_title(ENERGY_LABELS[cut], fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.margins(y=0.13)
    fig.suptitle(f"uds GEN-final AllShowers species | {n_events:,} unweighted events", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(folder / "overview_mean_multiplicity.png", dpi=135)
    plt.close(fig)


def make_figure(title):
    fig, axs = plt.subplots(3, 4, figsize=(15, 9.5))
    fig.suptitle(title, fontsize=13)
    return fig, axs


def finish_figure(fig, path):
    # Less layout work than constrained_layout for 12 axes on 24 figures.
    fig.tight_layout(rect=(0, 0, 1, 0.95), pad=0.8, w_pad=0.8, h_pad=1.0)
    fig.savefig(path, dpi=125)
    plt.close(fig)


def save_detailed_plots(folder):
    for region in REGIONS:
        for cut in DETAIL_CUTS:
            fig, axs = make_figure(
                f"Multiplicity | {region} | {ENERGY_LABELS[cut]} | {n_events:,} events")
            for ax, (pdg, name) in zip(axs.flat, SPECIES.items()):
                hist = multiplicity_hist((region, cut, pdg))
                xs = sorted(hist)
                ys = [hist[x] / n_events for x in xs]
                ax.bar(xs, ys, width=0.8, color="C0", edgecolor="none")
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.5 / n_events)
                ax.set_xlim(-0.7, max(3.7, xs[-1] + 0.7))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
                ax.set_title(name, loc="left", fontsize=10)
                ax.set_xlabel("N particles / event", fontsize=9)
                ax.set_ylabel("Fraction of events", fontsize=9)
                ax.tick_params(labelsize=8)
                annotate_panel(ax, region, cut, pdg)
            finish_figure(fig, folder / f"multiplicity_{region}_{cut}.png")

            for variable, edges in BINS.items():
                fig, axs = make_figure(
                    f"GEN {variable} | {region} | {ENERGY_LABELS[cut]} | {n_events:,} events")
                # Share one useful energy scale across the 12 species in a figure.
                energy_max = 260.0
                if variable == "energy":
                    nonzero_bins = [np.flatnonzero(kinematics[(region, cut, pdg, variable)])
                                    for pdg in SPECIES]
                    last = max((int(b[-1]) for b in nonzero_bins if len(b)), default=0)
                    energy_max = min(260.0, max(30.0, 10 * ceil(edges[last + 1] / 10) + 10))
                for ax, (pdg, name) in zip(axs.flat, SPECIES.items()):
                    counts = kinematics[(region, cut, pdg, variable)]
                    ax.stairs(counts, edges, color="C0", linewidth=1.1)
                    if variable == "energy":
                        ax.set_xlim(0, energy_max)
                        if counts.max() > 10:
                            ax.set_yscale("log")
                    elif variable == "theta":
                        ax.set_xlim(0, 180)
                    else:
                        ax.set_xlim(-180, 180)
                    ax.set_title(name, loc="left", fontsize=10)
                    ax.set_xlabel({"energy": "GEN E [GeV]",
                                   "theta": "GEN momentum theta [deg]",
                                   "phi": "GEN momentum phi [deg]"}[variable], fontsize=9)
                    ax.set_ylabel("Particles / bin", fontsize=9)
                    ax.tick_params(labelsize=8)
                    annotate_panel(ax, region, cut, pdg, n_plotted=int(counts.sum()))
                finish_figure(fig, folder / f"{variable}_{region}_{cut}.png")


def print_results(folder, read_seconds, plot_seconds):
    print(f"\n=== GEN study: {n_events:,} unweighted events ===")
    print(f"{'Selection':<22} {'N particles':>13} {'Mean/event':>12} {'Events >=1':>13}")
    for cut in ENERGY_CUTS:
        for region in REGIONS:
            total = sum(particle_totals[(region, cut, pdg)] for pdg in SPECIES)
            count_events = events_with_any[(region, cut)]
            print(f"{cut + ' / ' + region:<22} {total:>13,} {total/n_events:>12.3f} "
                  f"{count_events/n_events:>12.1%}")

    print("\nE >= 10 GeV: per-species counts (N) and means per event")
    print(f"{'Particle':<10} {'All N':>10} {'All mean':>10} {'Barrel N':>11} "
          f"{'Barrel mean':>12} {'Barrel/All':>11}")
    for pdg, name in SPECIES.items():
        all_n = particle_totals[("all_angles", "E10", pdg)]
        barrel_n = particle_totals[("barrel", "E10", pdg)]
        print(f"{name:<10} {all_n:>10,} {all_n/n_events:>10.3f} "
              f"{barrel_n:>11,} {barrel_n/n_events:>12.3f} "
              f"{barrel_n/all_n if all_n else 0:>10.1%}")
    print(f"\nReading + event analysis: {read_seconds:.1f} s")
    print(f"Writing summaries + making plots: {plot_seconds:.1f} s")
    print(f"Output: {folder}")


def main():
    global n_events
    files = sorted(INPUT_DIR.glob("E250-SetA.P2f-lcfi_z_*.edm4hep.root"))
    if len(files) != EXPECTED_FILES:
        raise RuntimeError(f"Found {len(files)} files in {INPUT_DIR}; "
                           f"expected {EXPECTED_FILES}. Check conversion.")
    if not set(DETAIL_CUTS).issubset(ENERGY_CUTS):
        raise ValueError(f"Unknown DETAIL_CUTS: {DETAIL_CUTS}")

    tag = ("full" if MAX_EVENTS_PER_FILE is None else
           "test_1k" if MAX_EVENTS_PER_FILE == 1000 else f"test_{MAX_EVENTS_PER_FILE}")
    folder = OUTPUT_DIR / f"{tag}_v2"  # Keep earlier plots untouched.
    folder.mkdir(parents=True, exist_ok=True)
    processed_files = []

    t_read_start = perf_counter()
    for idx, filename in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {filename.name}", flush=True)
        start = perf_counter()
        reader = get_reader(str(filename))
        n_file = 0
        collection_name = None
        for event in reader.get("events"):
            if MAX_EVENTS_PER_FILE is not None and n_file >= MAX_EVENTS_PER_FILE:
                break
            if collection_name is None:
                available = event.getAvailableCollections()
                if "MCParticles" in available:
                    collection_name = "MCParticles"
                elif "MCParticle" in available:
                    collection_name = "MCParticle"
                else:
                    raise RuntimeError(f"No MC particle collection in {filename}")
            analyse_event(event.get(collection_name))
            n_file += 1
            n_events += 1
            if n_file % 10000 == 0:
                print(f"  {n_file:,} events", flush=True)
        read_sec = perf_counter() - start
        del reader
        elapsed = perf_counter() - start
        processed_files.append((filename.name, n_file, f"{elapsed:.1f}", f"{read_sec:.1f}"))
        print(f"  {n_file:,} events in {elapsed:.1f} s", flush=True)

    if not n_events:
        raise RuntimeError("No events processed")
    read_seconds = perf_counter() - t_read_start
    t_plot_start = perf_counter()
    save_csvs(folder, processed_files)
    save_overview(folder)
    save_detailed_plots(folder)
    plot_seconds = perf_counter() - t_plot_start

    with (folder / "run_info.txt").open("w") as out:
        out.write(f"Files: {len(files)}\nEvents: {n_events}\n"
                  f"MAX_EVENTS_PER_FILE: {MAX_EVENTS_PER_FILE}\n"
                  f"DETAIL_CUTS: {DETAIL_CUTS}\n"
                  f"GEN-final particles (all PDGs): {n_final}\n"
                  f"GEN-final particles (12 species): {n_supported}\n"
                  f"Invalid energy: {invalid_energy}\n"
                  f"Undefined momentum direction: {undefined_direction}\n"
                  f"Out-of-range histogram fills: {dict(out_of_plot_range)}\n"
                  f"Reading/analysis time [s]: {read_seconds:.1f}\n"
                  f"Plot/CSV time [s]: {plot_seconds:.1f}\n"
                  "GEN status=1; E cuts: none, >=5 GeV, >=10 GeV.\n"
                  "GEN momentum barrel: 40<=theta<=140 deg, excluding "
                  "[C_k-0.51, C_k+8.5] deg at C_k=-157.5+45*k (k=0..7).\n"
                  "All events unweighted; no cross-section/polarisation weighting.\n")
    print_results(folder, read_seconds, plot_seconds)


if __name__ == "__main__":
    main()
