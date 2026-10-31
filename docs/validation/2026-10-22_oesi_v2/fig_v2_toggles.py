"""Four combinations of the revised scheme (OES on/off x CU on/off), p_phi = 2. 2026-10-22."""
import math, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from oesi import zeta_v2
NK, hK = 11186, 0.04
combos = [(False, False, "BAREI", "0.5", ":"), (False, True, "BARECUI", "tab:blue", "--"),
          (True, False, "OESI", "tab:orange", "-."), (True, True, "OESCUI", "tab:red", "-")]
PHI = np.geomspace(0.05, math.e, 200); MART = np.exp(PHI/math.e)
for oes, cui, lab, *_ in combos:
    z = np.array([zeta_v2(1, f, hK, NK, oes, cui) for f in PHI])
    print(f"{lab:10s} max |rest/Martin - 1| over 0.05..e: {100*np.max(np.abs(z/MART-1)):.1f}%")
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ph = np.geomspace(0.01, 10, 300)
ax[0].plot(ph, np.exp(ph/math.e), "k-", lw=2.2, label=r"Martin $e^{\phi/e}$")
for oes, cui, lab, c, ls in combos:
    ax[0].plot(ph, [zeta_v2(1, f, hK, NK, oes, cui) for f in ph], color=c, ls=ls, lw=1.4, label=lab)
ax[0].set(xscale="log", yscale="log", xlabel=r"$\phi$", ylabel=r"$\zeta_0/\zeta_Z$", title=fr"at rest, $h^*_K$={hK}, $N_K$={NK}")
ax[0].legend(fontsize=8)
E = np.geomspace(1, math.sqrt(NK), 400)
for a, f in zip(ax[1:], [0.01, 1.0]):
    for oes, cui, lab, c, ls in combos:
        a.plot(E, [zeta_v2(e, f, hK, NK, oes, cui) for e in E], color=c, ls=ls, lw=1.4, label=lab)
    a.set(xscale="log", yscale="log", xlabel=r"$E = R/R_0$", ylabel=r"$\zeta/\zeta_Z$", title=fr"$\phi$={f}")
fig.tight_layout(); fig.savefig("zeta_v2_toggles.png", dpi=150); fig.savefig("zeta_v2_toggles.pdf")
