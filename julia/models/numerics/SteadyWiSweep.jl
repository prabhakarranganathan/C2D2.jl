"""
    SteadyWiSweep.jl

Direct Wi sweeps for the steady model pack (STYEXT / PMF): solve the closure
state Z = (X, Y) at each prescribed Wi with `newton_fixed_Wi`, warm-starting
from the previous Wi and bisecting the Wi step when Newton fails.

This complements the PAC manifold builder (`StyextManifold.build_manifold`),
which places points adaptively along the (Y, μ) curve. The fixed-Wi sweep is
the natural driver for pole-free flows (planar mixed flow with r ≥ 1, e.g.
simple shear), where the technical report (Sec. 3.2.2) proves a unique
admissible steady state at every Wi, and it is what a Wi-grid comparison
against reference data needs.

Public API
----------
  sweep_fixed_Wi(ctx, Wi_values, solver_opts; kw...) -> Vector{StyextPoint}
  write_pmf_summary_csv(path, traj)   — PMF/shear CSV (flow-frame + strain-frame)
"""
module SteadyWiSweep

using Printf

using ..StyextModelPack: StyextMPCtx, pack_ztilde, unpack_ztilde,
                          eval_state_obs, get_Wi_pole
using ..StyextNewton:    SolverOpts, SolverStatus, newton_fixed_Wi
using ..StyextManifold:  StyextPoint
using ..StyextFlows:     wi_shear

export sweep_fixed_Wi, write_pmf_summary_csv

# ── One continuation leg with bisection ───────────────────────────────────────

"""
    _advance(ctx, X, Y, Wi_from, Wi_to, opts, max_bisect)
        -> (X, Y, mu, Wi_pole, FY_rel, n_it, ok)

Move the closure state from Wi_from (solution (X, Y)) to Wi_to. On a Newton
failure the leg is bisected (depth-first) until it converges or the total
number of bisections exceeds `max_bisect`.
"""
function _advance(ctx::StyextMPCtx, X::Vector{Float64}, Y::Float64,
                  Wi_from::Float64, Wi_to::Float64,
                  opts::SolverOpts, max_bisect::Int)
    Na      = ctx.Na
    Xc, Yc  = copy(X), Y
    muc     = NaN
    cur     = Wi_from
    n_it    = 0
    n_bis   = 0
    Wi_pole = NaN
    FY_rel  = NaN
    pending = [Wi_to]

    while !isempty(pending)
        target = pending[end]
        z0 = pack_ztilde(Xc, Yc, 0.0, Na)
        z, st = newton_fixed_Wi(ctx, z0, target, opts)
        n_it += st.newton_iters
        if st.converged && !st.min_denom_fail
            Xc, Yc, muc = unpack_ztilde(z, Na)
            cur     = target
            Wi_pole = st.Wi_pole
            FY_rel  = st.FY_rel
            pop!(pending)
        else
            n_bis += 1
            if n_bis > max_bisect || (target - cur) <= 1e-12 * max(abs(target), 1.0)
                return Xc, Yc, muc, Wi_pole, FY_rel, n_it, false
            end
            push!(pending, 0.5 * (cur + target))
        end
    end
    return Xc, Yc, muc, Wi_pole, FY_rel, n_it, true
end

# ── Sweep driver ──────────────────────────────────────────────────────────────

"""
    sweep_fixed_Wi(ctx, Wi_values, solver_opts;
                   include_equilibrium = true, max_bisect = 20,
                   stop_on_fail = true, verbose = false) -> Vector{StyextPoint}

Solve the steady closure state at each Wi in `Wi_values` (sorted ascending;
Wi = λ₁,₀ × principal strain rate, the report's convention, so for simple shear
Wi = λ₁,₀γ̇/2). Starts from equilibrium (X = Q₀², Y = E₀²) and warm-starts each
Wi from the previous solution.  `ctx` and `solver_opts` come from
`load_styext_context`; the `[sweep]` grid of the TOML is `params.Wi_sweep`.

Example
-------
    params, ctx, solver_opts, man_opts = load_styext_context("shear_example.toml")
    traj = sweep_fixed_Wi(ctx, params.Wi_sweep, solver_opts; X_stop = man_opts.X_stop)
    write_pmf_summary_csv("summary_sweep.csv", traj)

Termination: a point whose Y reaches ctx.Ymax (the max_frac_stretch_sq·E∞²
cap) ends the sweep; a leg that fails after `max_bisect` bisections ends it
when `stop_on_fail` (otherwise the Wi value is skipped with a warning).

Status strings: "eq", "FWi-cnv" (converged), "FWi-fail" (recorded with the last
converged state; also when `stop_on_fail = false`), "Y_stop", "X_stop" (last
converged point when max(X) ≥ `X_stop`, the FENE-PM/P closure pole).
"""
function sweep_fixed_Wi(ctx::StyextMPCtx, Wi_values::AbstractVector{<:Real},
                        opts::SolverOpts;
                        include_equilibrium::Bool = true,
                        max_bisect::Int = 20,
                        stop_on_fail::Bool = true,
                        X_stop::Float64 = Inf,
                        verbose::Bool = false)::Vector{StyextPoint}
    Na  = ctx.Na
    Wis = sort(unique(Float64.(Wi_values)))
    all(w -> isfinite(w) && w >= 0.0, Wis) || error("sweep_fixed_Wi: Wi_values must be finite and ≥ 0")
    filter!(w -> w > 0.0, Wis)

    X = Na > 0 ? fill(ctx.Q0_sq, Na) : Float64[]
    Y = ctx.Ymin
    traj = StyextPoint[]
    idx  = 0

    if include_equilibrium
        idx += 1
        push!(traj, StyextPoint(idx, copy(X), Y, 0.0, eval_state_obs(ctx, X, Y, 0.0),
                                get_Wi_pole(ctx, X, Y), 0, 0.0, 0.0, "eq"))
    end

    Wi_prev = 0.0
    for Wi in Wis
        Xn, Yn, mun, Wi_pole, FY_rel, n_it, ok =
            _advance(ctx, X, Y, Wi_prev, Wi, opts, max_bisect)
        idx += 1
        if !ok
            if Yn >= ctx.Ymax * (1.0 - 1e-6)
                # The steady state at this Wi needs Y > Ymax (= max_frac_stretch_sq·E∞²):
                # the Y cap, not the solver, ends the sweep.
                @warn "[FWi] Wi=$Wi: steady state requires Y > Ymax=$(ctx.Ymax) (Y cap reached at Y=$Yn). Stopping sweep."
                push!(traj, StyextPoint(idx, Xn, Yn, mun, eval_state_obs(ctx, Xn, Yn, isnan(mun) ? 0.0 : mun),
                                        Wi_pole, n_it, Wi - Wi_prev, FY_rel, "Y_stop"))
                break
            end
            @warn "[FWi] Wi=$Wi: fixed-Wi Newton failed after $max_bisect bisections (last Wi_pole=$Wi_pole)."
            # Record the failure (state = last converged point) whether or not we stop,
            # so a sweep never silently drops a Wi value.
            push!(traj, StyextPoint(idx, Xn, Yn, mun, eval_state_obs(ctx, Xn, Yn, isnan(mun) ? 0.0 : mun),
                                    Wi_pole, n_it, Wi - Wi_prev, FY_rel, "FWi-fail"))
            stop_on_fail && break
            continue
        end
        X, Y = Xn, Yn
        obs = eval_state_obs(ctx, X, Y, mun)
        push!(traj, StyextPoint(idx, copy(X), Y, mun, obs, Wi_pole, n_it, Wi - Wi_prev, FY_rel, "FWi-cnv"))
        if verbose
            @printf("[FWi] k=%4d  Wi=% .6e  Y=% .6e  eta_p=% .6e  Psi1=% .6e  newton=%3d  FY_rel=% .3e\n",
                    idx, Wi, Y, obs.eta_p, obs.Psi1, n_it, FY_rel)
        end
        Wi_prev = Wi
        if Y >= ctx.Ymax * (1.0 - 1e-12)
            @warn "[FWi] Y=$Y reached Ymax=$(ctx.Ymax) at Wi=$Wi. Stopping sweep."
            break
        end
        if Na > 0 && isfinite(X_stop) && maximum(X) >= X_stop * (1.0 - 1e-12)
            @warn "[FWi] X=$(maximum(X)) reached X_stop=$X_stop (FENE closure at the pole) at Wi=$Wi. Stopping sweep."
            traj[end] = StyextPoint(traj[end].idx, traj[end].X, traj[end].Y, traj[end].mu, traj[end].obs,
                                    traj[end].Wi_pole, traj[end].newton_iters, traj[end].ds, traj[end].FY_rel, "X_stop")
            break
        end
    end
    return traj
end

# ── CSV writer ────────────────────────────────────────────────────────────────

"""
    write_pmf_summary_csv(path, traj)

Planar-mixed-flow / shear summary CSV (one row per point).

Columns
  idx, E2, Wi, Wi_shear,                       Wi = λ₁,₀ × principal strain rate; Wi_shear = 2·Wi = λ₁,₀γ̇
  eta_p, Psi1, Psi2,                           polymer shear viscosity (units c kBT λ₁,₀) and first/second
                                               normal-stress coefficients (units c kBT λ₁,₀²), flow frame
  tau_xx, tau_yy, tau_zz, tau_xy,              flow-frame τ/(c kBT), compressive-positive sign (−Σ(3fM−I))
  tau_11, tau_22, tau_12,                      strain-axes frame (τ_33 = tau_zz)
  etaE1, etaE2, Tr1, Tr2,                      strain-frame planar extensional viscosities and Trouton ratios
  Wi_pole, newton_iters, ds, FY_rel, status    solver diagnostics ("eq", "FWi-cnv", "FWi-fail", "Y_stop", "X_stop")
Full description: docs/output_reference.md.
"""
function write_pmf_summary_csv(path::AbstractString, traj::Vector{StyextPoint})
    open(path, "w") do io
        println(io, "idx,E2,Wi,Wi_shear,eta_p,Psi1,Psi2,tau_xx,tau_yy,tau_zz,tau_xy," *
                    "tau_11,tau_22,tau_12,etaE1,etaE2,Tr1,Tr2,Wi_pole,newton_iters,ds,FY_rel,status")
        for pt in traj
            o    = pt.obs
            half = 0.5 * (o.tau_xx + o.tau_yy)
            t11  = half + o.tau_xy
            t22  = half - o.tau_xy
            t12  = 0.5 * (o.tau_yy - o.tau_xx)
            @printf(io, "%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%g,%d,%g,%g,%s\n",
                    pt.idx, o.E2, o.Wi, wi_shear(o.Wi),
                    o.eta_p, o.Psi1, o.Psi2,
                    o.tau_xx, o.tau_yy, o.tau_zz, o.tau_xy,
                    t11, t22, t12,
                    o.etaE1, o.etaE2, o.Tr1, o.Tr2,
                    pt.Wi_pole, pt.newton_iters, pt.ds, pt.FY_rel, pt.status_str)
        end
    end
end

end # module SteadyWiSweep
