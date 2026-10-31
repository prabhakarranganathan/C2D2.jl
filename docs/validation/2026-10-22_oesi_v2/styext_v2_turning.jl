# STYEXT steady-extension
# curves Wi(E²) at h*K ∈ {0.025, 0.04}, NK = 5000, φ ∈ {0.01, 0.1, 1}, single-mode FENE-P, for
# the four v2 schemes (p_phi = 2), reporting turning points of Wi(E²) below the coil–stretch
# transition.  Writes curve_<scheme>_h<hK>_phi<phi>.csv and turning_points.md next to itself.
# Run from the repo root:  julia --project=julia docs/validation/2026-10-22_oesi_v2/styext_v2_turning.jl [q_oes]
# With a q_oes argument > 0 (2026-10-22 pm) only OESI/OESCUI are run, with that
# q_oes, and the outputs carry a _q<q> tag (turning_points_q<q>.md).
const _MODELS = joinpath(@__DIR__, "..", "..", "..", "julia", "models")
for f in ("kernel/Fene.jl","kernel/Drag.jl","kernel/Spectrum.jl","kernel/CoeffsRelax.jl","kernel/evals/Stress.jl",
          "kernel/evals/StateObs.jl","kernel/flows/StyextFlows.jl","kernel/modelpacks/StyextModelPack.jl",
          "numerics/StyextNewton.jl","numerics/StyextManifold.jl","front_end/StyextRunContext.jl")
    include(joinpath(_MODELS, f))
end
using .Fene, .Drag, .Spectrum, .CoeffsRelax, .Stress, .StateObs, .StyextFlows,
      .StyextModelPack, .StyextNewton, .StyextManifold, .StyextRunContext
using Printf
outdir = @__DIR__
const Q_OES = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.0
const TAG   = Q_OES > 0 ? "_q$(Q_OES)" : ""
function make_toml(interp, hK)
    path = joinpath(outdir, "styext_v2_$(interp)_h$(hK)$(TAG).toml")
    open(path, "w") do io
        print(io, """
[run]
label = "styext_v2_$(interp)"
[polymer]
Q0_sq = 1.0
kBT = 1.0
lambda0 = 1.0
NK = 5000
Nm = 1
hK_star = $(hK)
[concentration]
phi = 1.0
[model]
fe_model = "fenep"
drag_model = "c2d2"
draining_model = "partialdraining"
spectrum_model = "auto"
drag_interp = "$(interp)"
p_phi = 2.0
q_oes = $(Q_OES)
drag_lut_n_points = 4000
[flow]
kind = "styext"
alphas = [-0.5, -0.5, 1.0]
[limits]
Wi_max = 20.0
max_frac_stretch_sq = 0.999
[continuation]
max_steps = 40000
ds0 = 0.001
ds_min = 1e-10
ds_max = 0.5
[newton]
tol_res = 1e-8
max_newton = 15
target_newton = 10
target_offmanifold_ratio = 0.3
[newton_fixed_Y]
tol_FY_rel_hard = 100.0
""")
    end
    return path
end
# Turning points of Wi along the continuation path (E² increases monotonically along it).
# The coil–stretch transition is the *largest* drop of Wi (Wi_cs → Wi_sc); anything before it is
# a spurious turning point of the kind the D(A) revision is meant to remove.
function turning_points(Wi, E2)
    tp = Tuple{Symbol,Float64,Float64}[]        # (:max | :min, Wi, E2)
    for i in 2:length(Wi)-1
        Wi[i-1] < Wi[i] >= Wi[i+1] && push!(tp, (:max, Wi[i], E2[i]))
        Wi[i-1] > Wi[i] <= Wi[i+1] && push!(tp, (:min, Wi[i], E2[i]))
    end
    return tp
end
schemes = Q_OES > 0 ? ("OESI", "OESCUI") : ("BAREI", "BARECUI", "OESI", "OESCUI")
rows = String[]
for hK in (0.025, 0.04), interp in schemes
    toml = make_toml(interp, hK)
    for phi in (0.01, 0.1, 1.0)
        params, mp_ctx, solver_opts, man_opts = load_styext_context(toml; phi=phi)
        traj = build_manifold(mp_ctx, solver_opts, man_opts)
        Wi = [pt.obs.Wi for pt in traj]; E2 = [pt.obs.E2 for pt in traj]
        open(joinpath(outdir, "curve_$(interp)$(TAG)_h$(hK)_phi$(phi).csv"), "w") do io
            println(io, "Wi,E2")
            for (w, e) in zip(Wi, E2); @printf(io, "%.10g,%.10g\n", w, e); end
        end
        tp = turning_points(Wi, E2)
        # coil–stretch window: the max followed by the deepest min
        mins = [k for (k, t) in enumerate(tp) if t[1] === :min]
        if isempty(mins)
            cs_txt = "none (monotone Wi)"; pre = tp
        else
            ksc = mins[argmin([tp[k][2] for k in mins])]
            kcs = findlast(k -> tp[k][1] === :max && k < ksc, 1:length(tp))
            cs_txt = @sprintf("Wi_cs = %.4f (E² = %.3g) → Wi_sc = %.4f (E² = %.3g), ratio %.3f",
                              tp[kcs][2], tp[kcs][3], tp[ksc][2], tp[ksc][3], tp[kcs][2] / tp[ksc][2])
            pre = tp[1:kcs-1]
        end
        pre_txt = isempty(pre) ? "none" : join([@sprintf("%s Wi=%.5f @ E²=%.3g", t[1], t[2], t[3]) for t in pre], "; ")
        line = @sprintf("| %g | %s | %g | %g | %d | %s | %s |", hK, interp, Q_OES, phi, length(traj), cs_txt, pre_txt)
        push!(rows, line); println(line); flush(stdout)
    end
end
open(joinpath(outdir, "turning_points$(TAG).md"), "w") do io
    println(io, "# STYEXT Wi(E²) turning points, v2 schemes (p_phi = 2), single-mode FENE-P, NK = 5000\n")
    println(io, "| h*K | scheme | q_oes | φ | npts | coil–stretch window | turning points before the window |")
    println(io, "|---|---|---|---|---|---|---|")
    foreach(r -> println(io, r), rows)
end
println("wrote turning_points$(TAG).md")
