"""
check_region_vs_flag.py -- where does the geometric region disagree with MCParticle.isHandledByFastSim()?
Reads the CC3 REC file; pi0 daughter photons with E >= 10 GeV (status 1, parent pi0).
Prints the flagged fraction vs theta and vs phi-phase so the true acceptance can be read off directly.
    python3 check_region_vs_flag.py [n_events]      (default 10000)
"""
import sys, math
import numpy as np
from podio import root_io

CC3 = "/data/dust/user/alimuham/thesis/reco/tau-pi0-cc3-taureco-10k-events-test_REC.edm4hep.root"
N_EV = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
PHI0 = -157.5          # crack centres: PHI0 + 45 k

rows, n = [], 0
for ev in root_io.Reader(CC3).get("events"):
    if n >= N_EV:
        break
    n += 1
    for mc in ev.get("MCParticles"):
        if mc.getGeneratorStatus() != 1 or mc.getPDG() != 22 or mc.getEnergy() < 10.0:
            continue
        par = mc.getParents()
        if len(par) == 0 or par[0].getPDG() != 111:
            continue
        m = mc.getMomentum()
        p = math.sqrt(m.x**2 + m.y**2 + m.z**2)
        th = math.degrees(math.acos(max(-1, min(1, m.z / p))))
        ph = math.degrees(math.atan2(m.y, m.x))
        rows.append((mc.getEnergy(), th, ph, float(mc.isHandledByFastSim())))
a = np.array(rows)
E, th, ph, fl = a[:, 0], a[:, 1], a[:, 2], a[:, 3] > 0.5
print(f"events {n}, photons E>=10: {len(a)}, flagged {fl.sum()} ({100*fl.mean():.1f}%)")

print("\nflagged fraction vs theta (2.5 deg bins, >=5 photons):")
for lo in np.arange(0, 180, 2.5):
    m = (th >= lo) & (th < lo + 2.5)
    if m.sum() >= 5:
        print(f"  theta {lo:6.1f}-{lo+2.5:6.1f}: N={m.sum():5d}  flagged {100*fl[m].mean():5.1f}%")

bar = (th >= 40) & (th <= 140)
psi = (ph - PHI0) % 45.0
print("\nBARREL (theta 40-140): flagged fraction vs psi = (phi - C) mod 45 deg  (dead zone ~ 0% flagged)")
for lo in np.arange(0, 45, 2.0):
    m = bar & (psi >= lo) & (psi < lo + 2.0)
    if m.sum() >= 5:
        print(f"  psi {lo:4.1f}-{lo+2:4.1f}: N={m.sum():5d}  flagged {100*fl[m].mean():5.1f}%")

ec = ((th >= 12) & (th <= 23)) | ((th >= 157) & (th <= 168))
print(f"\nENDCAP (theta 12-23 / 157-168): N={ec.sum()}, flagged {100*fl[ec].mean() if ec.sum() else 0:.1f}%")
for lo in np.arange(0, 45, 5.0):
    m = ec & (psi >= lo) & (psi < lo + 5.0)
    if m.sum() >= 5:
        print(f"  psi {lo:4.1f}-{lo+5:4.1f}: N={m.sum():5d}  flagged {100*fl[m].mean():5.1f}%")

o = ~bar & ~ec
print(f"\nother theta (outside barrel/endcap windows): N={o.sum()}, flagged {fl[o].sum()}")
np.savez("region_vs_flag_photons.npz", E=E, theta=th, phi=ph, flag=fl)
print("saved region_vs_flag_photons.npz")
