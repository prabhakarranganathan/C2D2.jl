"""
    StyextNewton.jl

Newton + Armijo solvers for the steady-extension PAC system.

Mirrors MATLAB's
  styext_solve_PAC.m / styext_solve_fixed_Y_newton.m

Two entry points:
  newton_pac       — Newton on full ztilde = [X; Y; μ] with arc constraint
  newton_fixed_Y   — Newton on [X; μ] with Y pinned (fixed-Y fallback)

Both use the same Armijo backtracking logic and return a SolverStatus.

Public API
----------
  SolverOpts     — solver control parameters
  SolverStatus   — outcome record
  newton_pac     — PAC Newton driver
  newton_fixed_Y — fixed-Y Newton driver
"""
module StyextNewton

using LinearAlgebra: norm, dot, cond, lu, ldiv!

using ..StyextModelPack: StyextMPCtx, rj_pac, rj_fixed_Y, unpack_ztilde, pack_ztilde

export SolverOpts, SolverStatus, newton_pac, newton_fixed_Y

# ── Solver control parameters ─────────────────────────────────────────────────

"""
    SolverOpts

Control parameters for the Newton + Armijo solvers.

Fields
------
tol_res         : Newton convergence tolerance (‖F‖₂ < tol_res)
max_newton      : maximum Newton iterations (PAC)
max_newton_fy   : maximum Newton iterations (fixed-Y fallback)
min_denom_tol   : D_pα floor — abort if any denominator ≤ this
tol_J_cond      : J abort if rcond(J) < tol_J_cond (= 1/cond)
use_armijo      : enable Armijo backtracking
armijo_c1       : Armijo sufficient-decrease parameter (Wolfe c₁)
armijo_beta     : step-size reduction factor
max_bt          : maximum backtracking steps
tol_FY_rel_hard : FY_rel threshold → hard convergence failure
tol_FY_rel_soft : FY_rel threshold → soft convergence warning
target_newton   : target Newton count for adaptive step-size
target_rho      : target off-manifold ratio ρ for adaptive step-size
"""
Base.@kwdef struct SolverOpts
    tol_res          :: Float64 = 1e-8
    max_newton       :: Int     = 15
    max_newton_fy    :: Int     = 15
    min_denom_tol    :: Float64 = 0.0
    tol_J_cond       :: Float64 = 1e-12
    use_armijo       :: Bool    = true
    armijo_c1        :: Float64 = 1e-4
    armijo_beta      :: Float64 = 0.5
    max_bt           :: Int     = 30
    tol_FY_rel_hard  :: Float64 = 100.0
    tol_FY_rel_soft  :: Float64 = 1e-2
    target_newton    :: Int     = 10
    target_rho       :: Float64 = 0.3
end

# ── Solver outcome ────────────────────────────────────────────────────────────

"""
    SolverStatus

Outcome of a Newton solve.

Fields (mirrors MATLAB status struct)
------
converged        : ‖F‖ ≤ tol_res on exit
newton_iters     : number of Newton iterations performed
residual_norm    : ‖F‖₂ at exit
armijo_bt_total  : total backtracking steps taken
Wi_pole          : Wi_pole at the exit iterate
FY_residual      : |FY| at the exit iterate
FY_rel           : FY_rel = |FY| / max(|Y|, E0_sq) at exit
FY_rel_soft_fail : FY_rel > tol_FY_rel_soft
FY_rel_hard_fail : FY_rel > tol_FY_rel_hard
min_denom_fail   : min D_pα ≤ min_denom_tol (denominator failure)
J_fail           : Jacobian ill-conditioned
armijo_fail      : Armijo backtracking exhausted without acceptance
max_newton_fail  : iteration limit reached without convergence
"""
mutable struct SolverStatus
    converged        :: Bool
    newton_iters     :: Int
    residual_norm    :: Float64
    armijo_bt_total  :: Int
    Wi_pole          :: Float64
    FY_residual      :: Float64
    FY_rel           :: Float64
    FY_rel_soft_fail :: Bool
    FY_rel_hard_fail :: Bool
    min_denom_fail   :: Bool
    J_fail           :: Bool
    armijo_fail      :: Bool
    max_newton_fail  :: Bool
end

function _init_status()
    SolverStatus(false, 0, NaN, 0, NaN, NaN, NaN,
                 false, false, false, false, false, false)
end

# ── FY diagnostics helper ─────────────────────────────────────────────────────

function _fy_diag!(st::SolverStatus, F_vec::Vector{Float64},
                   z::Vector{Float64}, idx_FY::Int,
                   E0_sq::Float64, opts::SolverOpts)
    FY_abs = abs(F_vec[idx_FY])
    Y_val  = z[idx_FY]
    FY_ref = max(abs(Y_val), E0_sq)
    FY_rel = FY_ref > 0 ? FY_abs / FY_ref : Inf

    st.FY_residual = FY_abs
    st.FY_rel      = FY_rel

    if FY_rel > opts.tol_FY_rel_hard
        st.FY_rel_hard_fail = true
        st.FY_rel_soft_fail = true
    elseif FY_rel > opts.tol_FY_rel_soft
        st.FY_rel_soft_fail = true
    end
end

# ── Shared Newton loop ────────────────────────────────────────────────────────

"""
    _newton_loop!(z, builder, opts, max_it, status, E0_sq, idx_FY)

Generic Newton+Armijo loop.
  builder(z) -> (F, J, info)
  idx_FY : index of FY in F (for diagnostics)
"""
function _newton_loop!(z        :: Vector{Float64},
                       builder  :: Function,
                       opts     :: SolverOpts,
                       max_it   :: Int,
                       status   :: SolverStatus,
                       E0_sq    :: Float64,
                       idx_FY   :: Int)

    for it in 1:max_it
        F, J, info = builder(z)
        Fv   = vec(F)
        resn = norm(Fv)

        status.residual_norm = resn
        status.newton_iters  = it
        status.Wi_pole       = info.Wi_pole

        # Denominator check
        if info.min_denom <= opts.min_denom_tol
            status.min_denom_fail = true
            break
        end

        # Convergence check
        if resn <= opts.tol_res
            status.converged = true
            break
        end

        nF = length(Fv)
        size(J, 1) == nF && size(J, 2) == nF ||
            error("_newton_loop!: rj_* must return square J matching F")

        # Condition check (approximate using lu and cond)
        Jcond = try cond(J) catch; Inf end
        if !isfinite(Jcond) || Jcond > 1.0 / max(opts.tol_J_cond, 1e-16)
            status.J_fail = true
            break
        end

        d = J \ (-Fv)     # Newton step

        if !opts.use_armijo
            z .+= d
        else
            phi0        = 0.5 * dot(Fv, Fv)
            gradphi_dir = -dot(Fv, Fv)   # directional derivative along d = -J\F

            alpha    = 1.0
            accepted = false
            bt       = 0

            while bt < opts.max_bt
                bt += 1
                z_trial = z .+ alpha .* d

                F_trial, _, info_trial = builder(z_trial)
                Fv_trial = vec(F_trial)
                phi_trial = 0.5 * dot(Fv_trial, Fv_trial)

                if info_trial.min_denom <= opts.min_denom_tol
                    alpha *= opts.armijo_beta
                    continue
                end

                if phi_trial <= phi0 + opts.armijo_c1 * alpha * gradphi_dir
                    z .= z_trial
                    accepted = true
                    break
                else
                    alpha *= opts.armijo_beta
                end
            end

            status.armijo_bt_total += bt

            if !accepted
                status.armijo_fail = true
                break
            end
        end
    end

    if !status.converged && !status.min_denom_fail &&
       !status.armijo_fail && !status.J_fail &&
       status.newton_iters >= max_it
        status.max_newton_fail = true
    end

    # FY diagnostics at exit
    F_end, _, info_end = builder(z)
    Fv_end = vec(F_end)
    _fy_diag!(status, Fv_end, z, idx_FY, E0_sq, opts)
    status.Wi_pole = info_end.Wi_pole

    nothing
end

# ── PAC Newton ───────────────────────────────────────────────────────────────

"""
    newton_pac(ctx, z0, zprev, zpred, t_prev, opts)
             -> (z_new, status)

Newton + Armijo solver for the PAC system on ztilde = [X; Y; μ].

builder = z -> rj_pac(ctx, z, zprev, zpred, t_prev)
FY is at index Na+1 (second-to-last entry of F).
"""
function newton_pac(ctx    :: StyextMPCtx,
                    z0     :: AbstractVector{Float64},
                    zprev  :: AbstractVector{Float64},
                    zpred  :: AbstractVector{Float64},
                    t_prev :: AbstractVector{Float64},
                    opts   :: SolverOpts)

    z      = Vector{Float64}(z0)
    status = _init_status()

    builder = (zz) -> rj_pac(ctx, zz, zprev, zpred, t_prev)

    Na     = ctx.Na
    nF     = Na + 2
    idx_FY = Na + 1   # FY is penultimate in F = [FX(1:Na); FY; Farc]
    E0_sq  = ctx.Ymin

    _newton_loop!(z, builder, opts, opts.max_newton, status, E0_sq, idx_FY)

    return z, status
end

# ── Fixed-Y Newton ────────────────────────────────────────────────────────────

"""
    newton_fixed_Y(ctx, z0, Y_fixed, opts)
                 -> (z_new, status)

Newton + Armijo solver for the fixed-Y system on [X; μ].

builder = z -> rj_fixed_Y(ctx, z, Y_fixed)
FY is the last entry of F = [FX(1:Na); FY] (index Na+1).
"""
function newton_fixed_Y(ctx     :: StyextMPCtx,
                         z0      :: AbstractVector{Float64},
                         Y_fixed :: Float64,
                         opts    :: SolverOpts)

    Na = ctx.Na

    # Build a reduced state vector z_red = [X(1:Na); μ]  (length Na+1).
    # The Y component is pinned and excluded from the Newton iteration
    # to match the (Na+1)×(Na+1) system returned by rj_fixed_Y.
    z_red = Vector{Float64}(undef, Na + 1)
    if Na > 0
        z_red[1:Na] .= z0[1:Na]
    end
    z_red[Na + 1] = z0[Na + 2]   # μ slot (z0[Na+2] in full vector)

    status = _init_status()

    # Builder: reconstruct full ztilde for rj_fixed_Y, then return (F, J, info)
    builder = function(zr::Vector{Float64})
        z_full = Vector{Float64}(undef, Na + 2)
        if Na > 0
            z_full[1:Na] .= zr[1:Na]
        end
        z_full[Na + 1] = Y_fixed
        z_full[Na + 2] = zr[Na + 1]   # μ
        return rj_fixed_Y(ctx, z_full, Y_fixed)
    end

    idx_FY = Na + 1   # last entry of F = [FX(1:Na); FY]
    E0_sq  = ctx.Ymin

    _newton_loop!(z_red, builder, opts, opts.max_newton_fy, status, E0_sq, idx_FY)

    # Reconstruct the full output vector z = [X(1:Na); Y_fixed; μ]
    z = Vector{Float64}(undef, Na + 2)
    if Na > 0
        z[1:Na] .= z_red[1:Na]
    end
    z[Na + 1] = Y_fixed
    z[Na + 2] = z_red[Na + 1]

    return z, status
end

end # module StyextNewton
