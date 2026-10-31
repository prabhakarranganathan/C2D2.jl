# Julia twin of fig_v2_toggles.py (2026-10-22): the four v2 schemes from Drag.jl, p_phi = 2,
# NK = 11186, h*K = 0.04 — at rest vs Martin's e^{φ/e}, and ζ(E) at φ = 0.01 and φ = 1.
# Run from the repo root:  julia --project=julia docs/validation/2026-10-22_oesi_v2/fig_v2_toggles_jl.jl
using CairoMakie, LaTeXStrings
const _MODELS = joinpath(@__DIR__, "..", "..", "..", "julia", "models")
include(joinpath(_MODELS, "kernel", "Drag.jl")); using .Drag
const NK = 11186.0; const hK = 0.04; const E_max = sqrt(NK)
z(E, phi, di) = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=di, p_phi=2.0)[1]
combos = ((:barei, "BAREI", :gray40, :dot), (:barecui, "BARECUI", :dodgerblue3, :dash),
          (:oesi, "OESI", :darkorange2, :dashdot), (:oescui, "OESCUI", :firebrick, :solid))
PHI = exp.(range(log(0.05), 1.0; length=200))
for (di, lab, _, _) in combos
    r = [z(1.0, p, di) for p in PHI] ./ exp.(PHI ./ ℯ)
    println(rpad(lab, 10), " max |rest/Martin − 1| over 0.05..e: ", round(100 * maximum(abs.(r .- 1)); digits=1), "%")
end
fig = Figure(size=(1400, 420), fontsize=13)
ax0 = Axis(fig[1, 1]; xscale=log10, yscale=log10, xlabel=L"\phi", ylabel=L"\zeta_0/\zeta_Z",
           title=L"at rest, $h^*_K = 0.04$, $N_K = 11186$")
ph = exp.(range(log(0.01), log(10.0); length=300))
lines!(ax0, ph, exp.(ph ./ ℯ); color=:black, linewidth=2.2, label=L"Martin $e^{\phi/e}$")
for (di, lab, c, ls) in combos
    lines!(ax0, ph, [z(1.0, p, di) for p in ph]; color=c, linestyle=ls, linewidth=1.5, label=lab)
end
axislegend(ax0; position=:lt, labelsize=10)
E = exp.(range(0.0, log(E_max); length=400))
for (k, phi) in enumerate((0.01, 1.0))
    ax = Axis(fig[1, k+1]; xscale=log10, yscale=log10, xlabel=L"E = R/R_0", ylabel=L"\zeta/\zeta_Z",
              title=L"\phi = %$(phi)")
    for (di, lab, c, ls) in combos
        lines!(ax, E, [z(e, phi, di) for e in E]; color=c, linestyle=ls, linewidth=1.5, label=lab)
    end
    k == 2 && axislegend(ax; position=:lt, labelsize=10)
end
save(joinpath(@__DIR__, "zeta_v2_toggles_jl.png"), fig; px_per_unit=2)
save(joinpath(@__DIR__, "zeta_v2_toggles_jl.pdf"), fig)
println("wrote zeta_v2_toggles_jl.{png,pdf}")
