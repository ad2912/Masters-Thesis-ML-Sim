import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio import root_io

IN = "/data/dust/user/alimuham/thesis/InputFiles/"
FILES = {
    "Unfiltered": IN + "elec_pos_500kevents.edm4hep.root",
    "Filtered":   IN + "tau_pi0_10GeV_filtered_500kevents.edm4hep.root",
}
COLORS = {"Unfiltered": "#7f7f7f", "Filtered": "#1b7837"}
E_CUT = 10.0
PLOTS = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/photons-after-filter-test"

ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, default=None, help="max events per file (default: all)")
ap.add_argument("--log", action="store_true", help="log y-axis")
args = ap.parse_args()

tag = "full" if args.n is None else (f"{args.n // 1000}k" if args.n % 1000 == 0 else str(args.n))
outdir = os.path.join(PLOTS, f"input_photon_counts_{tag}")
os.makedirs(outdir, exist_ok=True)


def count(path, nmax):
    A, B = [], []
    for i, event in enumerate(root_io.Reader(path).get("events")):
        if nmax is not None and i >= nmax:
            break
        mcs = event.get("MCParticles")
        if i == 0:
            print("  Event 0 particles: index, PDG, genStatus, E [GeV]")
            for k, p in enumerate(mcs):
                print("   ", k, p.getPDG(), p.getGeneratorStatus(), round(p.getEnergy(), 3))
        a = b = 0
        for p in mcs:
            if p.getPDG() == 22 and p.getGeneratorStatus() == 1:
                a += 1
                if p.getEnergy() >= E_CUT:
                    b += 1
        A.append(a)
        B.append(b)
        if (i + 1) % 50000 == 0:
            print(f"  processed {i + 1} events", flush=True)
    return np.array(A), np.array(B)


data = {}
for name, path in FILES.items():
    print(f"\n{name}: {path}")
    A, B = count(path, args.n)
    data[name] = {"A": A, "B": B}
    np.savez(os.path.join(outdir, f"counts_{name.lower()}.npz"), A=A, B=B)
    print(f"  Events: {len(A)}")
    print(f"  Mean photons/event (GS=1, no E cut):     {A.mean():.4f}")
    print(f"  Mean photons/event (GS=1, E>={E_CUT:g} GeV):  {B.mean():.4f}")
    print(f"  Events with B == 0: {(B == 0).sum()} | max A: {A.max()} | max B: {B.max()}")

panels = [
    ("A", "Photons per event (status 1, no energy cut)", "photons_per_event_noCut.png"),
    ("B", r"Photons per event (status 1, $E \geq 10$ GeV)", "photons_per_event_E10.png"),
]
for key, xlabel, fname in panels:
    top = max(d[key].max() for d in data.values())
    bins = np.arange(-0.5, top + 1.5, 1.0)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, d in data.items():
        v = d[key]
        w = np.full(len(v), 1.0 / len(v))
        lab = f"{name}  (N={len(v):,}, mean = {v.mean():.2f})"
        ax.hist(v, bins=bins, weights=w, histtype="stepfilled", alpha=0.35, color=COLORS[name])
        ax.hist(v, bins=bins, weights=w, histtype="step", color=COLORS[name], linewidth=2, label=lab)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Fraction of events", fontsize=18)
    ax.tick_params(labelsize=15)
    if args.log:
        ax.set_yscale("log")
    ax.legend(fontsize=14, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close(fig)

print(f"\nSaved to {outdir}")
