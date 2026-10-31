# Regenerate the two OES-note comparison figures from C2D2.jl (Drag.jl), replacing the
# Python reference (oesi.py / fig_oesi.py / fig_oesi_cu.py) so the note can cite the code.
#   zeta_OESI_vs_RPI_jl.{pdf,png}    RPI (drag_interp = "RP") dashed vs OESI solid
#   zeta_OESCUI_vs_OESI_jl.{pdf,png} OESI dashed vs OESCUI (p_phi = 1.86) solid, Martin e^{φ/e} dotted
# Run from the repo root:  julia --project=julia docs/validation/2026-10-21_oesi/fig_oesi_jl.jl
using CairoMakie, LaTeXStrings
const _MODELS = joinpath(@__DIR__, "..", "..", "..", "julia", "models")
include(joinpath(_MODELS, "kernel", "Drag.jl")); using .Drag

const NK = 5000.0; const E_max = sqrt(NK); const P_PHI = 1.86
z(E, phi, hK, di) = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=di, p_phi=P_PHI)[1]

"OESI dilute/weak boundary φ_b(E): root in φ of φE³ = 1 + ln(E/φ) (bisection in ln φ, as oesi.py)."
function phi_dil_ws(E)
    E <= 1 && return 1.0
    f(lp) = exp(lp) * E^3 - 1 - log(E) + lp
    lo, hi = -60.0, 0.0
    for _ in 1:200; m = 0.5(lo + hi); f(m) > 0 ? (hi = m) : (lo = m); end
    exp(0.5(lo + hi))
end

Es   = [1.0, 3.0, 10.0, 30.0, E_max]
cols = [:black, RGBf(0.122, 0.467, 0.706), RGBf(0.173, 0.627, 0.173), RGBf(1.0, 0.498, 0.055), RGBf(0.839, 0.153, 0.157)]
phis = exp10.(range(-3, 2; length=600))
labE(e) = e == E_max ? L"E/E_0=\sqrt{N_K}" : latexstring("E/E_0=$(Int(e))")

function panel!(ax, hK, di_dash, di_solid; markers::Symbol)
    for (e, c) in zip(Es, cols)
        lines!(ax, phis, [z(e, p, hK, di_dash) for p in phis]; color=c, linestyle=:dash, linewidth=1.3)
        lines!(ax, phis, [z(e, p, hK, di_solid) for p in phis]; color=c, linewidth=2.2, label=labE(e))
        if markers === :oesi_vs_rpi && e > 1
            pb = phi_dil_ws(e)
            scatter!(ax, [pb], [z(e, pb, hK, :oesi)]; color=:white, strokecolor=c, strokewidth=1.3, markersize=9)
            scatter!(ax, [e^-3], [z(e, e^-3, hK, :rp)]; color=c, marker=:rect, markersize=7)
        end
        e < E_max && scatter!(ax, [e], [z(e, e, hK, di_solid)]; color=c, markersize=9)
    end
    if markers === :oescui_vs_oesi
        ph = phis[phis .< 4]
        lines!(ax, ph, exp.(ph ./ ℯ); color=RGBf(0.4, 0.4, 0.4), linestyle=:dot, linewidth=1.6)
    end
end

function make(fname, di_dash, di_solid, markers, ttl_extra, legend2)
    fig = Figure(size=(1100, 450), fontsize=13)
    axs = [Axis(fig[1, i]; xscale=log10, yscale=log10, xlabel=L"\phi = c/c^\ast",
                ylabel=(i == 1 ? L"\zeta/\zeta_\mathrm{Z}" : ""),
                title=latexstring("h^\\ast_K = $(hK),\\ N_K = $(Int(NK))$(ttl_extra)"),
                xminorticksvisible=true, xminorgridvisible=false, yminorticksvisible=true,
                xminorticks=IntervalsBetween(9), yminorticks=IntervalsBetween(9))
           for (i, hK) in enumerate((0.17, 0.025))]
    for (ax, hK) in zip(axs, (0.17, 0.025))
        panel!(ax, hK, di_dash, di_solid; markers)
        xlims!(ax, 1e-3, 1e2)
    end
    linkyaxes!(axs...)
    axislegend(axs[1]; position=:lt, framevisible=true, labelsize=11)
    # second legend: line styles / markers
    el = legend2
    Legend(fig[1, 2], first.(el), last.(el); tellwidth=false, tellheight=false,
           halign=:left, valign=:top, margin=(8, 0, 0, 8), labelsize=11, framevisible=true)
    for ext in ("pdf", "png")
        save(joinpath(@__DIR__, "$fname.$ext"), fig; px_per_unit=2)
    end
    println("wrote $fname.{pdf,png}")
end

make("zeta_OESI_vs_RPI_jl", :rp, :oesi, :oesi_vs_rpi, "",
     [LineElement(color=:black, linestyle=:dash, linewidth=1.3) => "RPI",
      LineElement(color=:black, linewidth=2.2) => "OESI",
      MarkerElement(color=:black, marker=:rect, markersize=7) => L"RPI dilute/weak, $\phi E^3 = 1$",
      MarkerElement(color=:white, strokecolor=:black, strokewidth=1.3, marker=:circle, markersize=9) => "OESI dilute/weak",
      MarkerElement(color=:black, marker=:circle, markersize=9) => L"weak/strong, $\phi = E/E_0$"])
make("zeta_OESCUI_vs_OESI_jl", :oesi, :oescui, :oescui_vs_oesi, ",\\ p_\\phi = 1.86",
     [LineElement(color=:black, linestyle=:dash, linewidth=1.3) => "OESI",
      LineElement(color=:black, linewidth=2.2) => "OESCUI",
      LineElement(color=RGBf(0.4, 0.4, 0.4), linestyle=:dot, linewidth=1.6) => L"Martin, $e^{\phi/e}$",
      MarkerElement(color=:black, marker=:circle, markersize=9) => L"$\phi = E/E_0$"])
