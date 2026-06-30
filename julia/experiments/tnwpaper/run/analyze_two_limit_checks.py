#!/usr/bin/env python3
"""
analyze_two_limit_checks.py — sweep-level checks for the two-limit CaBER code
(TnW v4).  Uses the constant-drag generic sweeps:

  viscous : S03 (De_v = 0.01..100, generic_validation) + S08 (De_v = 300..3000)
  inertial: S05 (De_R = 0.01..100) + S07 (matched grid De_R = Oh*De_v)

Checks:
  A. regime-INVARIANT elastic plateau: Wi_e(matched De_v) inertial vs viscous,
     and plateau-level scaling vs phi.
  B. regime-DEPENDENT post-corner rise: d ln Wi_e / d ln De on the developed
     high-De branch (theory 4beta/3: 4/3 viscous, 8/9 inertial).

Only genuine minima (selection_reason == "deepest_min") count as plateau/Wi_e
points; fallback inflexions are reported separately and excluded from fits.

Outputs: printed fits + two_limit_checks.png in the generic_inertial folder.
Ko, 2026-06-11.
"""

import os, math, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", "..", ".."))
OUT_V = os.path.join(REPO, "julia/experiments/tnwpaper/outputs/generic_validation")
OUT_I = os.path.join(REPO, "julia/experiments/tnwpaper/outputs/generic_inertial")

OH   = 0.005
PHIS = [("0p01", 0.01), ("0p1", 0.1), ("0p5", 0.5), ("1", 1.0)]

SETS = {           # regime -> list of (root, folder template)
    "viscous": [
        (OUT_V, "S03_caber_sweep_De_NK_3000_hsK_0p05_phi_{p}_drag_constant"),
        (OUT_I, "S08_caber_sweep_DeHi_NK_3000_hsK_0p05_phi_{p}_drag_constant"),
    ],
    "inertial": [
        (OUT_I, "S05_caber_inertial_sweep_De_NK_3000_hsK_0p05_phi_{p}_drag_constant"),
        (OUT_I, "S07_caber_inertial_matched_De_NK_3000_hsK_0p05_phi_{p}_drag_constant"),
    ],
}
BETA = {"viscous": 1.0, "inertial": 2.0 / 3.0}


def _read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_regime_phi(regime, ptok):
    """Merge Wi_e + selection_reason across the sweep folders for one phi.

    Returns sorted list of dicts: {De, De_v, Wi_e, reason}.
    De_v is the viscous Deborah of the cell (De_R/Oh for inertial)."""
    rows = {}
    for root, tmpl in SETS[regime]:
        d = os.path.join(root, tmpl.format(p=ptok))
        wie = _read_csv(os.path.join(d, "Wi_e_vs_De.csv"))
        fvd = _read_csv(os.path.join(d, "wi_features_vs_De.csv"))
        reason = {}
        for r in fvd:
            sv = float(r["sweep_value"])
            # prefer the min row's reason if both min+max selected
            if r.get("selection_reason") == "deepest_min" or sv not in reason:
                reason[sv] = r.get("selection_reason", "?")
        for r in wie:
            de = float(r["De"])
            rows[de] = {
                "De": de,
                "De_v": de / OH if regime == "inertial" else de,
                "Wi_e": float(r["Wi_e"]),
                "reason": reason.get(de, "?"),
            }
    return [rows[k] for k in sorted(rows)]


def fit_loglog(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return np.nan, np.nan
    c = np.polyfit(np.log10(x[ok]), np.log10(y[ok]), 1)
    return c[0], 10 ** c[1]


def main():
    data = {}
    for regime in SETS:
        for ptok, phi in PHIS:
            data[(regime, phi)] = load_regime_phi(regime, ptok)

    # ── full tables ──────────────────────────────────────────────────────────
    print("=== merged Wi_e tables (m = deepest_min, f = fallback_inflexion) ===")
    for (regime, phi), rows in data.items():
        tag = {"deepest_min": "m", "fallback_inflexion": "f"}
        line = ", ".join(f"{r['De']:g}:{r['Wi_e']:.4g}{tag.get(r['reason'],'?')}"
                         for r in rows)
        print(f"  {regime:8s} phi={phi:<5g}  {line}")

    # ── Check A: regime-invariant plateau, De_v -> 0 limit ──────────────────
    # plateau estimate = Wi_e of the LOWEST-De_v deepest_min cell; flagged as
    # converged when the two lowest cells agree to < 3%.
    print("\n=== Check A: plateau Wi_e in the De_v -> 0 limit ===")
    plat = {}
    for (regime, phi), rows in data.items():
        sel = sorted((r for r in rows if r["reason"] == "deepest_min"),
                     key=lambda r: r["De_v"])
        if not sel:
            continue
        lv = sel[0]["Wi_e"]
        conv = (len(sel) >= 2 and
                abs(sel[1]["Wi_e"] / lv - 1.0) < 0.03)
        plat[(regime, phi)] = lv
        plat[(regime, phi, "conv")] = conv
    for ptok, phi in PHIS:
        v = plat.get(("viscous", phi))
        i = plat.get(("inertial", phi))
        if v and i:
            cv = "conv" if plat.get(("viscous", phi, "conv")) else "NOT conv"
            ci = "conv" if plat.get(("inertial", phi, "conv")) else "NOT conv"
            print(f"  phi={phi:<5g}  viscous {v:.4g} ({cv})"
                  f"   inertial {i:.4g} ({ci})   ratio {i/v:.3f}")
        else:
            print(f"  phi={phi:<5g}  viscous {v}   inertial {i}")
    for regime in SETS:
        pp = [(phi, plat[(regime, phi)]) for _, phi in PHIS
              if (regime, phi) in plat]
        if len(pp) >= 2:
            s, _ = fit_loglog([p for p, _ in pp], [w for _, w in pp])
            print(f"  {regime:8s} plateau-vs-phi exponent (all phi) = {s:.3f}"
                  f"   [dilute theory -4/3]")

    # ── Check B: post-corner rise (min features, developed branch) ──────────
    print("\n=== Check B: post-corner rise d ln Wi_e/d ln De"
          "  [theory 4b/3: viscous 1.333, inertial 0.889] ===")
    for (regime, phi), rows in data.items():
        sel = [r for r in rows if r["reason"] == "deepest_min"]
        if len(sel) < 3:
            continue
        # developed branch = top decade of De among min cells that rose
        # at least 1.5x above the plateau level
        base = plat.get((regime, phi))
        if base is None:
            continue
        risen = [r for r in sel if r["Wi_e"] > 1.5 * base]
        if len(risen) >= 2:
            s, _ = fit_loglog([r["De"] for r in risen],
                              [r["Wi_e"] for r in risen])
            des = [r["De"] for r in risen]
            print(f"  {regime:8s} phi={phi:<5g}  slope = {s:.3f}"
                  f"   (De {min(des):g}..{max(des):g}, {len(risen)} pts)")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = {0.01: "tab:blue", 0.1: "tab:orange", 0.5: "tab:green",
              1.0: "tab:red"}
    for (regime, phi), rows in data.items():
        ls, mk = ("-", "o") if regime == "viscous" else ("--", "s")
        col = colors[phi]
        xm = [r["De"] for r in rows if r["reason"] == "deepest_min"]
        ym = [r["Wi_e"] for r in rows if r["reason"] == "deepest_min"]
        xf = [r["De"] for r in rows if r["reason"] != "deepest_min"]
        yf = [r["Wi_e"] for r in rows if r["reason"] != "deepest_min"]
        axes[0].loglog(xm, ym, ls + mk, ms=4, color=col,
                       label=f"{regime[:4]} φ={phi:g}")
        axes[0].loglog(xf, yf, mk, ms=4, color=col, mfc="none", alpha=0.5)
        # matched-De_v axis
        xmv = [r["De_v"] for r in rows if r["reason"] == "deepest_min"]
        axes[1].loglog(xmv, ym, ls + mk, ms=4, color=col)
    axes[0].set_xlabel("De (De$_v$ viscous / De$_R$ inertial)")
    axes[0].set_ylabel("Wi$_e$")
    axes[0].set_title("Pipkin curves — filled+line = deepest_min,\n"
                      "open = fallback inflexion (excluded)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].set_xlabel("matched De$_v$ = De$_R$/Oh (inertial) or De$_v$ (viscous)")
    axes[1].set_ylabel("Wi$_e$")
    axes[1].set_title("Regime invariance test: same De$_v$, both regimes\n"
                      "solid = viscous, dashed = inertial")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    png = os.path.join(OUT_I, "two_limit_checks.png")
    fig.savefig(png, dpi=160)
    print(f"\nFigure: {png}")


if __name__ == "__main__":
    main()
