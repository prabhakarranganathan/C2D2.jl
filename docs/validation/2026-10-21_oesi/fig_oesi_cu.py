import math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from oesi import *
NK = 5000; Emax = math.sqrt(NK)
Es = [1, 3, 10, 30, Emax]
cols = ["k", "tab:blue", "tab:green", "tab:orange", "tab:red"]
phis = np.logspace(-3, 2, 600)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for a, hK in zip(ax, [0.17, 0.025]):
    for e, c in zip(Es, cols):
        zo = [zeta_oesi(e, p, hK, NK) for p in phis]
        zc = [zeta_oescui(e, p, hK, NK) for p in phis]
        lab = r"$E/E_0=\sqrt{N_K}$" if e == Emax else rf"$E/E_0={e:g}$"
        a.loglog(phis, zo, "--", color=c, lw=1.1)
        a.loglog(phis, zc, "-", color=c, lw=1.8, label=lab)
        if e < Emax:
            a.loglog([e], [zeta_oescui(e, e, hK, NK)], "o", ms=4.5, color=c)
    ph = phis[phis < 4]
    a.loglog(ph, np.exp(ph/math.e), ":", color="0.4", lw=1.4)
    a.set_title(rf"$h^\ast_K={hK}$, $N_K={NK}$, $p_\phi=1.86$")
    a.set_xlabel(r"$\phi=c/c^\ast$"); a.set_xlim(1e-3, 1e2)
ax[0].set_ylabel(r"$\zeta/\zeta_\mathrm{Z}$")
ax[0].legend(fontsize=8, loc="upper left")
ax[1].plot([], [], "k--", lw=1.1, label="OESI")
ax[1].plot([], [], "k-", lw=1.8, label="OESCUI")
ax[1].plot([], [], ":", color="0.4", lw=1.4, label=r"Martin, $e^{\phi/e}$")
ax[1].plot([], [], "ko", ms=4.5, label=r"$\phi=E/E_0$")
ax[1].legend(fontsize=8, loc="upper left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"zeta_OESCUI_vs_OESI.{ext}", dpi=200)
for hK in [0.17, 0.025]:
    print(hK, "rest phi=0.3,1,2,e:", [round(zeta_oescui(1, p, hK, NK), 3) for p in [0.3, 1, 2, math.e]],
          "Martin:", [round(math.exp(p/math.e), 3) for p in [0.3, 1, 2, math.e]])
    for e in Es[1:]:
        r = [zeta_oescui(e, p, hK, NK)/zeta_oesi(e, p, hK, NK) for p in phis]
        i = int(np.argmax(r)); print("  E=%.1f max CU/OESI %.2f at phi %.3g" % (e, r[i], phis[i]))
