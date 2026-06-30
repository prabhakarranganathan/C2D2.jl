"""
    StyextManifold.jl

Pseudo-arclength continuation (PAC) manifold builder for steady extension.

Mirrors MATLAB's styext_pac_mu_manifold.m.

Algorithm:
  1. Seed at equilibrium: (X=Q₀², Y=E₀², μ=0)
  2. First step: fixed-Y Newton at Y₁ (geometric step from E₀² toward E∞²)
  3. Subsequent steps:
       a. Secant predictor from the two previous accepted points
       b. PAC corrector (Newton + Armijo on full ztilde = [X; Y; μ])
       c. Fallback to fixed-Y corrector if PAC fails or diverges
       d. Adaptive step-size based on Newton cost + off-manifold ratio ρ

Termination conditions:
  - Y ≥ Y_stop (= max_frac_stretch_sq · E∞²)
  - Wi ≥ Wi_max
  - FY_rel > tol_FY_rel_hard
  - ds < ds_min with no acceptance

Output: a Vector{StyextPoint} and a CSV summary file.

Public API
----------
  ManifoldOpts       — step-size and termination parameters
  StyextPoint        — one point on the manifold
  build_manifold     — run PAC, return Vector{StyextPoint}
  write_summary_csv  — write output CSV
"""
module StyextManifold

using LinearAlgebra: norm
using Printf

using ..StyextModelPack: StyextMPCtx, pack_ztilde, unpack_ztilde,
                          eval_state_obs, get_Wi_pole
using ..StyextNewton:    SolverOpts, SolverStatus, newton_pac, newton_fixed_Y
using ..StateObs:        C2D2Obs

export ManifoldOpts, StyextPoint, build_manifold, write_summary_csv

# ── Manifold options ──────────────────────────────────────────────────────────

"""
    ManifoldOpts

Step-size and termination controls for the PAC manifold builder.
"""
Base.@kwdef struct ManifoldOpts
    ds0              :: Float64 = 1e-3
    ds_min           :: Float64 = 1e-8
    ds_max           :: Float64 = 1e3
    max_steps        :: Int     = 5000
    Wi_max           :: Float64 = Inf
    tol_FY_rel_hard  :: Float64 = 100.0
    target_newton    :: Int     = 10
    target_rho       :: Float64 = 0.3
    verbose          :: Bool    = false
end

# ── Manifold point ────────────────────────────────────────────────────────────

"""
    StyextPoint

One accepted point on the steady-extension manifold.
"""
struct StyextPoint
    idx          :: Int
    X            :: Vector{Float64}
    Y            :: Float64
    mu           :: Float64
    obs          :: C2D2Obs
    Wi_pole      :: Float64
    newton_iters :: Int
    ds           :: Float64
    FY_rel       :: Float64
    status_str   :: String
end

# ── Secant predictor ──────────────────────────────────────────────────────────

function _predictor(ctx  :: StyextMPCtx,
                    Xm2, Ym2::Float64, mum2::Float64,
                    Xm1, Ym1::Float64, mum1::Float64,
                    ds   :: Float64)
    Na = ctx.Na
    z1 = pack_ztilde(Xm2, Ym2, mum2, Na)
    z2 = pack_ztilde(Xm1, Ym1, mum1, Na)

    dz = z2 .- z1
    n  = norm(dz)
    if n > 0
        dz ./= n
    else
        # Default direction: increase Y
        fill!(dz, 0.0)
        dz[Na + 1] = 1.0
    end

    zpred = z2 .+ ds .* dz

    Xp, Yp, mup = unpack_ztilde(zpred, Na)

    # Safety clamps
    if Na > 0
        for i in eachindex(Xp)
            Xp[i] = isfinite(Xp[i]) ? max(Xp[i], 0.0) : 0.0
        end
    end
    Yp  = isfinite(Yp)  ? clamp(Yp,  ctx.Ymin, ctx.Ymax) : ctx.Ymin
    mup = isfinite(mup) ? clamp(mup, -20.0, 50.0)         : 0.0

    return Xp, Yp, mup
end

# ── Corrector: PAC → fixed-Y fallback ────────────────────────────────────────

function _corrector(ctx         :: StyextMPCtx,
                    Xp, Yp::Float64, mup::Float64,
                    Xprev, Yprev::Float64, muprev::Float64,
                    solver_opts :: SolverOpts)

    Na    = ctx.Na
    zpred = pack_ztilde(Xp,    Yp,    mup,    Na)
    zprev = pack_ztilde(Xprev, Yprev, muprev, Na)

    v         = zpred .- zprev
    n         = norm(v)
    use_pac   = n > 0.0

    z_init_fy = copy(zprev)   # seed for fixed-Y (updated to PAC iterate if PAC runs)

    # ── 1. PAC attempt ────────────────────────────────────────────────────────
    if use_pac
        t_prev = v ./ n
        z_pac, stat_pac = newton_pac(ctx, zpred, zprev, zpred, t_prev, solver_opts)

        pac_clean = stat_pac.converged &&
                    !stat_pac.min_denom_fail &&
                    !stat_pac.J_fail &&
                    !stat_pac.armijo_fail &&
                    !stat_pac.max_newton_fail &&
                    !stat_pac.FY_rel_soft_fail

        if pac_clean
            Xk, Yk, muk = unpack_ztilde(z_pac, Na)
            return Xk, Yk, muk,
                   stat_pac.Wi_pole, stat_pac.FY_rel,
                   true, stat_pac.newton_iters, "PAC-cnv"
        end

        z_init_fy = z_pac   # PAC iterate is often a better seed for fixed-Y
    end

    # ── 2. Fixed-Y fallback ───────────────────────────────────────────────────
    Y_fixed         = Yp
    z_init_fy[Na+1] = Y_fixed

    z_fy, stat_fy = newton_fixed_Y(ctx, z_init_fy, Y_fixed, solver_opts)
    Xk, _, muk = unpack_ztilde(z_fy, Na)
    Yk = Y_fixed

    J_hard      = stat_fy.J_fail     && stat_fy.FY_rel_soft_fail
    armijo_hard = stat_fy.armijo_fail && stat_fy.FY_rel_soft_fail
    hard_fail   = J_hard || armijo_hard || stat_fy.min_denom_fail || stat_fy.max_newton_fail
    soft_ok     = !stat_fy.converged && !hard_fail && !stat_fy.FY_rel_soft_fail

    status_str = (stat_fy.converged || soft_ok) ? "FY-cnv" : "FY-dnc"

    return Xk, Yk, muk,
           stat_fy.Wi_pole, stat_fy.FY_rel,
           (stat_fy.converged || soft_ok), stat_fy.newton_iters, status_str
end

# ── Adaptive step-size ────────────────────────────────────────────────────────

function _adapt_ds(Xp, Yp::Float64, mup::Float64,
                   Xk, Yk::Float64, muk::Float64,
                   Xprev, Yprev::Float64, muprev::Float64,
                   n_it       :: Int,
                   status_str :: String,
                   ds         :: Float64,
                   opts       :: ManifoldOpts,
                   Na         :: Int)

    accepted = (status_str != "FY-dnc")

    zpred = pack_ztilde(Xp,    Yp,    mup,    Na)
    zcorr = pack_ztilde(Xk,    Yk,    muk,    Na)
    zprev = pack_ztilde(Xprev, Yprev, muprev, Na)

    d_pred = zpred .- zprev
    d_corr = zcorr .- zpred
    rho    = norm(d_corr) / (norm(d_pred) + eps(Float64))

    p_newton = Float64(n_it) / max(Float64(opts.target_newton), eps(Float64))
    p_rho    = rho / max(opts.target_rho, eps(Float64))
    severity = max(p_newton, p_rho)

    if !accepted
        return max(ds * 0.5, opts.ds_min), false
    end

    if severity > 2.0
        ds_next = max(ds * 0.5, opts.ds_min)
    elseif severity < 0.5
        ds_next = min(ds * 1.3, opts.ds_max)
    else
        ds_next = ds
    end

    return ds_next, true
end

# ── Main manifold builder ─────────────────────────────────────────────────────

"""
    build_manifold(ctx, solver_opts, man_opts) -> Vector{StyextPoint}

Run the PAC manifold builder and return all accepted points.
"""
function build_manifold(ctx         :: StyextMPCtx,
                        solver_opts :: SolverOpts,
                        man_opts    :: ManifoldOpts)::Vector{StyextPoint}

    Na        = ctx.Na
    Y_start   = ctx.Ymin
    Y_stop    = ctx.Ymax    # max_frac_stretch_sq * Einf_sq (encoded at build time)
    ds        = man_opts.ds0
    max_steps = man_opts.max_steps
    max_steps >= 2 || error("build_manifold: ManifoldOpts.max_steps must be ≥ 2")
    seed_eps = 1e-4

    traj = StyextPoint[]

    # ── Point 1: equilibrium ─────────────────────────────────────────────────
    X0  = Na > 0 ? fill(ctx.Q0_sq, Na) : Float64[]
    Y0  = Y_start
    mu0 = 0.0

    obs0     = eval_state_obs(ctx, X0, Y0, mu0)
    Wi_pole0 = get_Wi_pole(ctx, X0, Y0)

    push!(traj, StyextPoint(1, X0, Y0, mu0, obs0, Wi_pole0, 0, ds, 0.0, "eq"))

    # ── Point 2: fixed-Y seed step ────────────────────────────────────────────
    ratio       = Y_stop / Y_start
    step_factor = ratio^(1.0 / (max_steps - 1))
    Y1          = Y_start * step_factor

    # Perturb to break symmetry
    Xseed = Na > 0 ? X0 .* (1.0 + seed_eps) : Float64[]
    Yseed = Y1
    museed = mu0 + seed_eps

    # Force fixed-Y by using pred == prev → zero secant tangent
    Xk, Yk, muk, Wi_pole_k, FY_rel_k, _, n_it_k, status_k =
        _corrector(ctx, Xseed, Yseed, museed,
                       Xseed, Yseed, museed, solver_opts)

    obs1 = eval_state_obs(ctx, Xk, Yk, muk)
    push!(traj, StyextPoint(2, Xk, Yk, muk, obs1, Wi_pole_k, n_it_k, ds, FY_rel_k, status_k))

    if man_opts.verbose
        @printf("[PAC] k=%4d  Y=% .6e  Wi=% .6e  mu=% .6e  ds=% .3e  newton=%3d  FY_rel=% .3e  %s\n",
                2, Yk, obs1.Wi, muk, ds, n_it_k, FY_rel_k, status_k)
    end

    ds, _ = _adapt_ds(Xseed, Yseed, museed, Xk, Yk, muk,
                       X0, Y0, mu0, n_it_k, status_k, ds, man_opts, Na)

    Xm2, Ym2, mum2 = X0,  Y0,  mu0
    Xm1, Ym1, mum1 = Xk,  Yk,  muk

    reason = "max_steps"
    k = 3

    while k <= max_steps
        accepted      = false
        stepsize_fail = false

        local Xk_acc, Yk_acc, muk_acc, Wi_pole_acc, FY_rel_acc, n_it_acc, status_acc

        while !accepted && !stepsize_fail
            # ── Predictor ────────────────────────────────────────────────────
            Xp, Yp, mup = _predictor(ctx, Xm2, Ym2, mum2, Xm1, Ym1, mum1, ds)

            # ── Corrector ────────────────────────────────────────────────────
            Xk_acc, Yk_acc, muk_acc, Wi_pole_acc, FY_rel_acc, _, n_it_acc, status_acc =
                _corrector(ctx, Xp, Yp, mup, Xm1, Ym1, mum1, solver_opts)

            old_ds = ds
            ds, accepted = _adapt_ds(Xp, Yp, mup, Xk_acc, Yk_acc, muk_acc,
                                     Xm1, Ym1, mum1, n_it_acc, status_acc, ds, man_opts, Na)

            if old_ds <= man_opts.ds_min * (1.0 + 1e-12) &&
               ds     <= man_opts.ds_min * (1.0 + 1e-12) && !accepted
                stepsize_fail = true
            end
        end

        if stepsize_fail
            @warn "[PAC] ds ≤ ds_min=$(man_opts.ds_min) with no acceptance. Stopping."
            reason = "ds_min"
            break
        end

        obs_k = eval_state_obs(ctx, Xk_acc, Yk_acc, muk_acc)

        push!(traj, StyextPoint(k, Xk_acc, Yk_acc, muk_acc,
                                obs_k, Wi_pole_acc,
                                n_it_acc, ds, FY_rel_acc, status_acc))

        if man_opts.verbose
            @printf("[PAC] k=%4d  Y=% .6e  Wi=% .6e  mu=% .6e  ds=% .3e  newton=%3d  FY_rel=% .3e  %s\n",
                    k, Yk_acc, obs_k.Wi, muk_acc, ds, n_it_acc, FY_rel_acc, status_acc)
        end

        # ── Shift history ─────────────────────────────────────────────────────
        Xm2, Ym2, mum2 = Xm1, Ym1, mum1
        Xm1, Ym1, mum1 = Xk_acc, Yk_acc, muk_acc

        # ── Termination checks ────────────────────────────────────────────────
        if Yk_acc >= Y_stop * (1.0 - 1e-12)
            reason = "Y_stop"
            break
        end
        if isfinite(man_opts.Wi_max) && obs_k.Wi >= man_opts.Wi_max
            @warn "[PAC] Wi=$(obs_k.Wi) ≥ Wi_max=$(man_opts.Wi_max). Stopping."
            reason = "Wi_max"
            break
        end
        if isfinite(FY_rel_acc) && FY_rel_acc > man_opts.tol_FY_rel_hard
            @warn "[PAC] FY_rel=$(FY_rel_acc) > $(man_opts.tol_FY_rel_hard). Stopping."
            reason = "FY_rel_hard"
            break
        end

        k += 1
    end

    return traj
end

# ── CSV writer ────────────────────────────────────────────────────────────────

"""
    write_summary_csv(path, traj)

Write manifold trajectory to CSV (standard STYEXT summary format).

Columns: idx,E2,Wi,etaE1,etaE2,Tr1,Tr2,tau_xx,tau_yy,tau_zz,Wi_pole,newton_iters,ds,FY_rel,status
"""
function write_summary_csv(path::AbstractString, traj::Vector{StyextPoint})
    open(path, "w") do io
        println(io, "idx,E2,Wi,etaE1,etaE2,Tr1,Tr2,tau_xx,tau_yy,tau_zz,Wi_pole,newton_iters,ds,FY_rel,status")
        for pt in traj
            o = pt.obs
            @printf(io, "%d,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%d,%g,%g,%s\n",
                    pt.idx,
                    o.E2, o.Wi,
                    o.etaE1, o.etaE2,
                    o.Tr1, o.Tr2,
                    o.tau_xx, o.tau_yy, o.tau_zz,
                    pt.Wi_pole,
                    pt.newton_iters,
                    pt.ds,
                    pt.FY_rel,
                    pt.status_str)
        end
    end
end

end # module StyextManifold
