"""Reference implementation of the RPI and OESI drag schemes (2026-10-21).
RPI  = Drag.jl _drag_c2d2 (drag_interp "RP"), ported line by line.
OESI = overlap-enhanced shielding interpolation (c2d2_OES.tex, Sec. 3).
Both return zeta/zeta_Z for stretch E = R/R0 >= 1 and phi = c/c*.
"""
import math
A_INF = 3*math.pi/1.6366            # non-draining limit of alpha

def alpha(hK, n):
    """Kirkwood-Riseman shielding factor of a blob of n Kuhn segments (Drag.jl form)."""
    if math.isinf(n):
        return A_INF
    aK = math.sqrt(math.pi)*hK
    x = math.sqrt(n)
    pr = 0.9119 - 0.0191/x + 0.1009/x**2
    psr = 1.6366 - 1.1434/x - 0.4521/x**2
    return 3*math.pi*2*aK/max(pr/x + 2*aK*psr, 1e-300)

def zeta_rpi(E, phi, hK, NK):
    aZ = alpha(hK, NK); Emax = math.sqrt(NK); tiny = 2.220446049250313e-16   # eps(Float64), as in Drag.jl
    if phi > Emax:
        xiC, nC = 1/Emax, NK; zbC = alpha(hK, 1.0)*xiC/aZ
    elif phi > 1:
        nC = phi**2; xiC = 1/phi; zbC = alpha(hK, NK/nC)*xiC/aZ
    else:
        nC, xiC, zbC = 1.0, 1.0, 1.0
    xiT = 1/E; nT = min(max(E**2, 1.0), NK); A = E/xiT
    if E > phi:
        zbT = alpha(hK, NK/nT)*xiT/aZ
        phiT = max(phi*E*xiT**2, tiny)
        B = 1 + math.sqrt(max(math.log(1/phiT), 0.0)/phiT)
        Y = A - 1
        J = 1 + (0.5*Y + (2/math.e)*Y**2)/max(A, tiny)
        G1 = 1 + math.log(max((A + J*(B-1))/max(A+B, tiny), tiny))
        G2 = 1 + math.log(max(B/max(B + 1/max(A, tiny), tiny), tiny))
        G = max(G1/max(G2, tiny), tiny)
        return max(zbT*nT/G, tiny)
    return max(zbC*nC, tiny)

def chi_over_xit(phiT):
    """Regularised Mackaplow-Shaqfeh screening length, valid for phi_t <= 1."""
    if phiT <= 0:
        return math.inf
    return math.sqrt((1 + math.log(1/phiT))/phiT)

def J_new(A):
    return 1 + (2/math.e)*(A-1)**2/A

def zeta_oesi(E, phi, hK, NK, detail=False, psi_form="phi"):
    aZ = alpha(hK, NK); Emax = math.sqrt(NK)
    # psi_form "phi": psi = min(phi, 1) (adopted); "phiE": psi = min(phi*E, 1), overlap fraction of the stretched chain
    psi = min(phi*E, 1.0) if psi_form == "phiE" else min(phi, 1.0)
    if E > phi:                                    # dilute or weakly screened: tension-blob core
        nT = min(max(E**2, 1.0), NK); xiT = 1/E
        a = alpha(hK, NK/nT); a = a + (A_INF - a)*psi
        cx = chi_over_xit(phi/E)
        A = min(E**2, cx)
        z = (a/aZ)*xiT*nT/(1 + math.log(J_new(A)))
        reg = "dilute" if E**2 <= cx else "weak"
    else:                                          # strongly screened: correlation-blob core, non-draining
        if phi > Emax:
            z = (A_INF/aZ)*Emax
        else:
            z = (A_INF/aZ)*phi
        reg = "strong"
    return (z, reg) if detail else z

def phi_dil_ws_new(E):
    """phi on the OESI dilute/weak boundary, phi E^3 = 1 + ln(E/phi) (bisection in ln phi)."""
    if E <= 1: return 1.0
    f = lambda lp: math.exp(lp)*E**3 - 1 - math.log(E) + lp
    lo, hi = -60.0, 0.0
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if f(mid) > 0: hi = mid
        else: lo = mid
    return math.exp(0.5*(lo+hi))

def zeta_oescui(E, phi, hK, NK, p=1.86, cap=True):
    """OESI with a Churchill-Usagi blend on top: [zeta_w^p + zeta_s^p]^(1/p).
    zeta_w: OESI tension-blob drag, continued above the weak/strong boundary with A = 1
            (i.e. (a_inf/aZ)*E once phi >= E).
    zeta_s: non-draining strongly screened drag continued to all phi, (a_inf/aZ)*min(phi, sqrt(NK)).
    cap:    the result is capped at the Rouse drag (a_inf/aZ)*sqrt(NK).
    """
    aZ = alpha(hK, NK); Emax = math.sqrt(NK)
    if E > phi:
        zw = zeta_oesi(E, phi, hK, NK)
    else:
        nT = min(max(E**2, 1.0), NK)
        zw = (A_INF/aZ)*(1/E)*nT
    zs = (A_INF/aZ)*min(phi, Emax)
    z = (zw**p + zs**p)**(1/p)
    if cap:   # Rouse drag of N_K non-draining Kuhn segments (effective h* = a_inf/(6 pi^1.5) ~ 0.17)
        z = min(z, (A_INF/aZ)*Emax)
    return z


# ---------------------------------------------------------------------------------------------
# 2026-10-22 revision: monotone Batchelor denominator, p_phi = 2, independent OES / CU toggles.
# ---------------------------------------------------------------------------------------------
P_PHI_OESCUI = 2.0

def D_new(A):
    """Monotone replacement for 1 + ln J(A). sqrt(A) for A <= 4, 2 + ln(A/4) above.
    D(1) = 1; C1 at A = 4 (value 2, slope 1/4). In the dilute regime (A = E^2) the shape
    factor E/D is 1 for E <= 2 and x/(1 + ln x), x = E/2, above: monotone, and Batchelor's
    E/ln(k E^2) with k = e^2/4 at large E."""
    return math.sqrt(A) if A <= 4.0 else 2.0 + math.log(A/4.0)

def zeta_v2(E, phi, hK, NK, oes=True, cui=True, p=P_PHI_OESCUI, detail=False, q=0.0):
    """zeta/zeta_Z for the revised scheme. oes: overlap-enhanced shielding on/off. cui: capped
    Churchill-Usagi blend on/off. (True, True) = OESCUI, (True, False) = OESI,
    (False, False) = bare C2D2 with the new dilute/weak boundary and D(A), (False, True) = bare + CU."""
    aZ = alpha(hK, NK); Emax = math.sqrt(NK)
    nT = min(max(E**2, 1.0), NK); xiT = 1/E
    aT = alpha(hK, NK/nT)
    if oes:   # q: stretch-delayed dressing, psi = min(phi/E^q, 1) (2026-10-22 pm; q = 1/4 adopted)
        aT = aT + (A_INF - aT)*min(phi/E**q, 1.0)
    nC = min(max(phi, 1.0)**2, NK)
    aS = A_INF if oes else alpha(hK, NK/nC)             # correlation-blob (strong) shielding
    zs = (aS/aZ)*min(phi, Emax)                          # strong drag, continued to all phi
    if E > phi:                                          # dilute or weak
        cx = chi_over_xit(phi/E)
        A = min(E**2, cx)
        zw = (aT/aZ)*xiT*nT/D_new(A)
        reg = "dilute" if E**2 <= cx else "weak"
    else:                                                # strong: tension-blob drag continued with A = 1
        zw = (aT/aZ)*xiT*nT
        reg = "strong"
    if cui:
        z = (zw**p + zs**p)**(1/p)
        z = min(z, (aS if oes else alpha(hK, 1.0))/aZ*Emax)   # Rouse cap
    else:
        z = zw if E > phi else zs
    return (z, reg) if detail else z
