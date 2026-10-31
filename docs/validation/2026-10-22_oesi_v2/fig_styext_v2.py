"""Steady uniaxial extension E^2 vs Wi for the four v2 schemes (STYEXT curves), p_phi = 2. 2026-10-22."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = [("BAREI", "0.5", ":"), ("BARECUI", "tab:blue", "--"), ("OESI", "tab:orange", "-."), ("OESCUI", "tab:red", "-")]
fig, ax = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
for i, h in enumerate(["0.025", "0.04"]):
    for j, f in enumerate(["0.01", "0.1", "1.0"]):
        a = ax[i, j]
        for s, c, ls in S:
            d = np.loadtxt(f"curve_{s}_h{h}_phi{f}.csv", delimiter=",", skiprows=1)
            a.plot(d[:, 0], d[:, 1], color=c, ls=ls, lw=1.3, label=s)
        a.set(xscale="log", yscale="log", xlim=(0.03, 3), title=fr"$h^*_K$={h}, $\phi$={float(f):g}")
        if i == 1: a.set_xlabel(r"Wi (scaled with rest relaxation time)")
        if j == 0: a.set_ylabel(r"$E^2 = \langle R^2\rangle/R_0^2$")
ax[0, 0].legend(fontsize=8)
fig.suptitle(r"Steady uniaxial extension, single-mode FENE-P, $N_K$=5000, $p_\phi$=2")
fig.tight_layout(); fig.savefig("styext_v2_curves.png", dpi=150); fig.savefig("styext_v2_curves.pdf")
