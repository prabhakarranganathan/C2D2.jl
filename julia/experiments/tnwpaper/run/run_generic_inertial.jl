# run_generic_inertial.jl — export + run the two-block generic inertial sweeps
# (U_R = 0.01 and U_R = 1, constant + c2d2 drag) from Plan_Generic.
# Self-contained: exports the plan TOMLs, then runs every sweep.

include(joinpath(@__DIR__, "export_plan_from_xlsx.jl"))
include(joinpath(@__DIR__, "run_exptl_comparison.jl"))   # make_prod_sweeps, run_*_sweep, helpers

# Optional ARGS[1] selects one block slug ("ur0p01" or "ur1"); default = both.
# Splitting per block lets a stiff cell time out its own job without blocking
# the other block.
blocks = (isempty(ARGS) || isempty(strip(String(ARGS[1])))) ? ["ur0p01", "ur1"] : [String(ARGS[1])]

# 1. (Re)export just the selected block(s) from Plan_Generic
plans = String[]
for b in blocks
    plan = joinpath(@__DIR__, "plans", "gen_$(b).toml")
    export_plan_from_xlsx("generic_inertial_$(b)", plan; sheet="Plan_Generic")
    push!(plans, plan)
end

# 2. Run every sweep in the selected plan(s) through make_prod_sweeps + run loop

for plan in plans
    println("\n########## PLAN: ", plan, " ##########"); flush(stdout)
    specs = make_prod_sweeps(plan)
    for (i, spec) in enumerate(specs)
        println("[", i, "/", length(specs), "] ", spec.folder, "  (", spec.kind, ")"); flush(stdout)
        t0 = time()
        try
            if startswith(spec.kind, "styext")
                phi_vals = _get_phi_values_from_toml(spec.input_toml)
                run_styext_sweep(spec.sweep_dir, spec.input_toml;
                                 phi_values=phi_vals, x_mode=_x_mode_from_kind(spec.kind), verbose=false)
            elseif startswith(spec.kind, "caber")
                de_vals = _get_De_values_from_toml(spec.input_toml)
                run_usyext_sweep(spec.sweep_dir, spec.input_toml;
                                 De_values=de_vals, x_mode=_x_mode_from_kind(spec.kind), verbose=false)
            end
            println("    sweep done in ", round(time()-t0; digits=1), "s"); flush(stdout)
        catch e
            println("SWEEP_FAILED ", spec.folder, " :: ", e); flush(stdout)
        end
    end
end
println("ALL_SWEEPS_DONE")
