import math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from oesi import *
plt.rcParams.update({"font.size": 10})

# ---------- regime map ----------
E = np.logspace(0, 2, 400)
phi_old = E**-3
phi_new = np.array([phi_dil_ws_new(e) for e in E])
fig, a = plt.subplots(figsize=(5.4, 4.4))
a.fill_betweenx(E, 1e-5, phi_old, color="0.90")
a.fill_betweenx(E, phi_old, phi_new, color="tab:orange", alpha=0.20)
a.loglog(phi_old, E, "k-", lw=1.8, label=r"RPI: $\phi E^3=1$")
a.loglog(phi_new, E, color="tab:orange", lw=2.2, label=r"OESI: $\phi E^3=1+\ln(E/\phi)$")
Ews = np.logspace(0, 2, 50)
a.loglog(Ews, Ews, color="tab:blue", lw=1.6, label=r"weak/strong: $E=\phi$")
a.loglog(1/E, E, "k:", lw=0.9, label=r"$\phi E=1$")
a.set_xlim(1e-5, 10); a.set_ylim(1, 100)
a.set_xlabel(r"$\phi=c/c^\ast$"); a.set_ylabel(r"$E/E_0$")
a.text(3e-5, 2.0, "dilute", fontsize=11)
a.text(1.5e-2, 6, "weakly\nscreened", fontsize=11)
a.text(1.6, 1.25, "strongly\nscreened", fontsize=9)
a.legend(fontsize=8, loc="upper right", framealpha=0.95)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"oesi_regimes.{ext}", dpi=200)

# ---------- drag comparison ----------
NK = 5000; Emax = math.sqrt(NK)
Es = [1, 3, 10, 30, Emax]
cols = ["k", "tab:blue", "tab:green", "tab:orange", "tab:red"]
phis = np.logspace(-3, 2, 600)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for a, hK in zip(ax, [0.17, 0.025]):
    for e, c in zip(Es, cols):
        zr = [zeta_rpi(e, p, hK, NK) for p in phis]
        zo = [zeta_oesi(e, p, hK, NK) for p in phis]
        lab = r"$E/E_0=\sqrt{N_K}$" if e == Emax else rf"$E/E_0={e:g}$"
        a.loglog(phis, zr, "--", color=c, lw=1.1)
        a.loglog(phis, zo, "-", color=c, lw=1.8, label=lab)
        if e > 1:
            pb = phi_dil_ws_new(e)
            a.loglog([pb], [zeta_oesi(e, pb, hK, NK)], "o", ms=4.5, mfc="white", mec=c)
            a.loglog([e**-3], [zeta_rpi(e, e**-3, hK, NK)], "s", ms=3.5, color=c)
        if e < Emax:
            a.loglog([e], [zeta_oesi(e, e, hK, NK)], "o", ms=4.5, color=c)
    a.set_title(rf"$h^\ast_K={hK}$, $N_K={NK}$")
    a.set_xlabel(r"$\phi=c/c^\ast$")
    a.set_xlim(1e-3, 1e2)
ax[0].set_ylabel(r"$\zeta/\zeta_\mathrm{Z}$")
ax[0].legend(fontsize=8, loc="upper left")
ax[1].plot([], [], "k--", lw=1.1, label="RPI"); ax[1].plot([], [], "k-", lw=1.8, label="OESI")
ax[1].plot([], [], "ks", ms=3.5, label=r"RPI dilute/weak, $\phi E^3=1$")
ax[1].plot([], [], "o", ms=4.5, mfc="white", mec="k", label="OESI dilute/weak")
ax[1].plot([], [], "ko", ms=4.5, label=r"weak/strong, $\phi=E/E_0$")
ax[1].legend(fontsize=8, loc="upper left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"zeta_OESI_vs_RPI.{ext}", dpi=200)

# ---------- diagnostics ----------
for hK in [0.17, 0.025]:
    aZ = alpha(hK, NK)
    print(f"hK={hK}: alphaZ={aZ:.4f}, A_INF/alphaZ={A_INF/aZ:.4f}")
    for e in Es:
        # continuity at boundaries
        pb = phi_dil_ws_new(e) if e > 1 else None
        if e > 1:
            l, r = zeta_oesi(e, pb*0.9999, hK, NK), zeta_oesi(e, pb*1.0001, hK, NK)
            print(f"  E={e:6.2f} dil/ws phi={pb:.3e}: {l:.5f} {r:.5f}")
        if e < Emax:
            l, r = zeta_oesi(e, e*0.99999, hK, NK), zeta_oesi(e, e*1.00001, hK, NK)
            print(f"  E={e:6.2f} w/s  phi={e:.3g}: {l:.5f} {r:.5f}")
    # monotonic in E at fixed phi?
    bad = []
    for p in [1e-3, 1e-2, 0.1, 0.3, 1, 3, 10]:
        Eg = np.logspace(0, math.log10(Emax), 2000)
        z = np.array([zeta_oesi(e, p, hK, NK) for e in Eg])
        d = np.diff(z)
        if (d < -1e-12).any():
            i = np.argmin(z); bad.append((p, Eg[np.argmax(d<0)], z.min()/z[0]))
    print("  non-monotonic in E (phi, first E of decrease, min/zeta0):", bad)
