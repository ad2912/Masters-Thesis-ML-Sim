if len(links_for_unmatched) == 0:
    ct = costheta(unmatched_mc)
    e  = unmatched_mc.getEnergy()
    # Exclude TYPE_C2 — same cut as diagnostic script
    if e < 1.0 or ct > 0.99:
        continue   # ← add this line, then append below
    energies_typeC.append(e)
    costh_typeC.append(ct)
