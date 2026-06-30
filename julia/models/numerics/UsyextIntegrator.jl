"""
    UsyextIntegrator.jl

Adaptive ETDRK2 integrator for unsteady extension (USYEXT).

Mirrors MATLAB's usyext_etd2_strain.m.

Algorithm
---------
Integrates the diagonal modal conformation tensors M ∈ ℝ^{Nm×3} forward in
Hencky strain (or dimensionless time in low-Wi regimes).  The integration
variable switches automatically based on Wi (hybrid mode).

Each accepted step uses an embedded ETD1/ETD2 pair:
  ETD1 (predictor / stage-a):  1st-order exponential integrator
  ETD2 (corrected):            2nd-order exponential integrator

The local error estimate is |M_ETD2 - M_ETD1| scaled by atol + rtol·|M|.
Step size is adapted using a PI-like controller (safety * errn^{-1/2}).

Output
------
  Vector{UsyextPoint} — one entry per sample strain, plus (possibly) a
  truncated entry if a termination condition is hit first.

Public API
----------
  TolOpts          — adaptive controller parameters
  IntegrationPlan  — compiled sampling schedule
  UsyextCtx        — polymer model context (kernel + scales)
  UsyextPoint      — one trajectory snapshot
  integrate_usyext — main driver; returns Vector{UsyextPoint}
  write_summary_csv — write CSV in reference format
"""
module UsyextIntegrator

using LinearAlgebra: dot
using Printf

using ..UsyextFlows:   ConstantWiFlow, CaberFlow, eval_flow
using ..CoeffsRelax:   CoeffsKernel, ClosureZ, eval_coeffs!
using ..Stress:        polymer_stress
using ..StateObs:      C2D2Obs

export TolOpts, IntegrationPlan, UsyextCtx, UsyextPoint,
       integrate_usyext, write_summary_csv, write_caber_diag_csv

# ── Tolerance / controller options ────────────────────────────────────────────

"""
    TolOpts

Adaptive step-size controller parameters for the USYEXT ETDRK2 integrator.
Mirrors MATLAB's S_ctx.tol struct.
"""
Base.@kwdef struct TolOpts
    rtol          :: Float64 = 1e-7
    atol          :: Float64 = 1e-9
    safety        :: Float64 = 0.9
    grow_max      :: Float64 = 2.0
    shrink_min    :: Float64 = 0.2
    dstrain_min   :: Float64 = 1e-8
    dstrain_max   :: Float64 = 5e-2
    dt_min        :: Float64 = 1e-10
    dt_max        :: Float64 = 1.0
    scale_floor   :: Float64 = 1e-14
    epsdot_floor  :: Float64 = 1e-16
    hit_eps       :: Float64 = 1e-12
    overshoot_eps :: Float64 = 1e-10
    max_attempts  :: Int     = 50_000_000
    max_wall_s    :: Float64 = Inf
end

# ── Integration plan ──────────────────────────────────────────────────────────

"""
    IntegrationPlan

Compiled sampling schedule.  Built by UsyextRunContext.

Fields
------
independent               : :strain | :time | :auto
sample_strains            : sorted vector of output strain values (includes 0 and strain_max)
dstrain_init              : initial trial step in strain
dt_init                   : initial trial step in time
Wi_min_strain_independent : Wi threshold below which time-mode is used (:auto only)
"""
struct IntegrationPlan
    independent               :: Symbol
    sample_strains            :: Vector{Float64}
    dstrain_init              :: Float64
    dt_init                   :: Float64
    Wi_min_strain_independent :: Float64
end

# ── Model context ─────────────────────────────────────────────────────────────

"""
    UsyextCtx

Runtime context for the USYEXT polymer model.

Holds the CoeffsKernel (drag/spectrum/FENE evaluators) and the derived
physical scales needed to compute observables.

Fields
------
kernel   : CoeffsKernel — FENE + drag LUT + spectrum
Sp       : Nm-vector of Rouse end-to-end weights
fe_model : :fenepme | :fenepm | :fenep
alphas   : NTuple{3} of principal flow-rate ratios
i_ext    : index of extensional axis (argmax alphas)
i_comp   : index of primary compressive axis (argmin alphas)
Nm       : number of modes
E0       : equilibrium coil size √(Q₀² Nm)  — normalization for drag/spectrum LUT
Ymax     : max_frac_stretch_sq · E∞²         — termination threshold
lambda0  : longest relaxation time
etas     : solvent viscosity
ckBT     : c · k_B · T
"""
struct UsyextCtx
    kernel   :: CoeffsKernel
    Sp       :: Vector{Float64}
    fe_model :: Symbol
    alphas   :: NTuple{3, Float64}
    i_ext    :: Int
    i_comp   :: Int
    Nm       :: Int
    E0       :: Float64
    Ymax     :: Float64
    lambda0  :: Float64
    etas     :: Float64
    ckBT     :: Float64
end

# ── Trajectory point ──────────────────────────────────────────────────────────

"""
    UsyextPoint

One snapshot on the USYEXT trajectory.

For CaBER runs the fields R, X, psi, N1_p, K_til carry the capillary flow state.
For constant-Wi runs those fields are NaN.
"""
struct UsyextPoint
    idx     :: Int
    estrain :: Float64
    t       :: Float64
    Wi      :: Float64
    De      :: Float64
    E2      :: Float64
    etaE1   :: Float64
    etaE2   :: Float64
    Tr1     :: Float64
    Tr2     :: Float64
    tau_xx  :: Float64
    tau_yy  :: Float64
    tau_zz  :: Float64
    R       :: Float64    # filament radius  (NaN for non-CaBER)
    X       :: Float64    # capillary geometry factor (NaN for non-CaBER)
    psi     :: Float64    # stress-balance psi  (NaN for non-CaBER)
    N1_p    :: Float64    # polymer first normal stress diff (NaN for non-CaBER)
    K_til   :: Float64    # K̃ = φ·Γ_eq·Uz   (NaN for non-CaBER)
    status  :: Symbol     # :ok | :terminated
end

# ── phi₁ and phi₂ functions ───────────────────────────────────────────────────

"""
    _phi1phi2(z) -> (phi1, phi2)

Robust evaluation of the ETD phi-functions element-wise on a matrix z.

  phi1(z) = (eᶻ - 1) / z
  phi2(z) = (eᶻ - 1 - z) / z²

5-term Taylor series used for |z| < 1e-2 to avoid catastrophic cancellation.
The direct formulas lose ~8 significant digits at |z| = 1e-4 (eᶻ ≈ 1 + z + …;
subtraction of nearly equal quantities). The threshold 1e-2 keeps max Taylor
truncation error below 1e-12 for both phi1 and phi2 (next term is z⁵/720 ≈ 1e-11
at z=0.01). A 1e-6 threshold is unsafe: |z| in (1e-6, 1e-2) uses the direct
formula and loses up to ~8 digits.
"""
function _phi1phi2(z::Matrix{Float64})
    phi1 = similar(z)
    phi2 = similar(z)
    @inbounds for i in eachindex(z)
        zi = z[i]
        if abs(zi) < 1e-2
            # Taylor: phi1(z) = 1 + z/2 + z²/6 + z³/24 + z⁴/120
            phi1[i] = 1.0 + zi*(0.5 + zi*(1.0/6.0 + zi*(1.0/24.0 + zi/120.0)))
            # Taylor: phi2(z) = 1/2 + z/6 + z²/24 + z³/120 + z⁴/720
            phi2[i] = 0.5 + zi*(1.0/6.0 + zi*(1.0/24.0 + zi*(1.0/120.0 + zi/720.0)))
        else
            ez = exp(zi)
            phi1[i] = (ez - 1.0) / zi
            phi2[i] = (ez - 1.0 - zi) / (zi * zi)
        end
    end
    return phi1, phi2
end

# ── Closure Z from Nm×3 M_diag ───────────────────────────────────────────────

"""
    _fill_closure_Z!(Z, M, ctx)

Fill the ClosureZ struct in-place from a 3-column M_diag (Nm×3: xx, yy, zz).

Computes:
  Z.Y  = Σ_p Sₚ · (M[p,1]+M[p,2]+M[p,3])
  Z.E  = √Y / E₀             (MATLAB-compatible normalisation)
  Z.Xa = auxiliary closure variables (fe_model-dependent)
"""
function _fill_closure_Z!(Z::ClosureZ,
                           M::Matrix{Float64},
                           ctx::UsyextCtx)
    Nm = ctx.Nm
    Sp = ctx.Sp
    Y        = 0.0
    sum_trM  = 0.0

    @inbounds for p in 1:Nm
        trM_p  = M[p,1] + M[p,2] + M[p,3]
        Y     += Sp[p] * trM_p
        sum_trM += trM_p
    end

    Z.Y = Y
    Z.E = sqrt(max(Y, 0.0)) / ctx.E0   # matches MATLAB normalisation r=√Y/E₀

    fe = ctx.fe_model
    if fe === :fenepm
        Z.Xa[1] = sum_trM / Nm
    elseif fe === :fenep
        @inbounds for p in 1:Nm
            Z.Xa[p] = M[p,1] + M[p,2] + M[p,3]
        end
    end
    # :fenepme → Xa is empty; nothing to set
    nothing
end

# ── State evaluation ──────────────────────────────────────────────────────────

"""
    _eval_state(ctx, flow, t, strain, M) -> (flow_state, theta_p, sigma_p, fp, Y)

Evaluate polymer micro-state and flow at (t, strain, M).

Returns COPIES of theta_p, sigma_p, fp (kernel.scratch is mutable and reused).
"""
function _eval_state(ctx::UsyextCtx, flow, t::Float64, strain::Float64,
                     M::Matrix{Float64})
    # Fill ClosureZ using the kernel's reusable Z_buf
    Z = ctx.kernel.Z_buf
    _fill_closure_Z!(Z, M, ctx)

    eval_coeffs!(ctx.kernel, Z)
    sc = ctx.kernel.scratch

    # Copy before the next eval_state call overwrites scratch
    theta_p = copy(sc.theta_p)
    sigma_p = copy(sc.sigma_p)
    fp      = copy(sc.fp)
    Y       = Z.Y

    # N1_p = -3 Σ_p fₚ (M_ext - M_comp)
    N1_p = 0.0
    @inbounds for p in 1:ctx.Nm
        N1_p -= 3.0 * fp[p] * (M[p, ctx.i_ext] - M[p, ctx.i_comp])
    end

    flow_state = eval_flow(flow, t, strain, N1_p)

    return flow_state, theta_p, sigma_p, fp, Y
end

# ── Observables from state ────────────────────────────────────────────────────

"""
    _make_obs(ctx, flow_state, M, fp, Y) -> UsyextPoint(...)

Compute extensional viscosity observables from a state snapshot.
Wi = 0 → etaE / Tr = NaN (USYEXT convention, unlike STYEXT which uses eps floor).
"""
function _make_obs(ctx::UsyextCtx,
                   flow_state,
                   M::Matrix{Float64},
                   fp::Vector{Float64},
                   Y::Float64,
                   idx::Int,
                   estrain::Float64,
                   t::Float64,
                   status::Symbol)

    tau = polymer_stress(M, fp)
    Wi  = flow_state.Wi
    De  = flow_state.De

    taus   = (tau.xx, tau.yy, tau.zz)
    tau_e  = taus[ctx.i_ext]
    tau_c1 = taus[ctx.i_comp]
    # second compressive axis
    i_comp2 = (ctx.i_comp == 1) ? (ctx.i_ext == 2 ? 3 : 2) : (ctx.i_ext == 1 ? 3 : 1)
    tau_c2  = taus[i_comp2]

    if abs(Wi) < 1e-15
        etaE1 = NaN;  etaE2 = NaN;  Tr1 = NaN;  Tr2 = NaN
    else
        # strain_rate = Wi/λ₀;  etaE = -(τ_ext - τ_comp)/(strain_rate·λ₀) = -(τ_ext-τ_comp)/Wi
        etaE1 = -(tau_e - tau_c1) / Wi
        etaE2 = -(tau_e - tau_c2) / Wi
        Tr1   = etaE1 * ctx.ckBT * ctx.lambda0 / (3.0 * ctx.etas)
        Tr2   = etaE2 * ctx.ckBT * ctx.lambda0 / (3.0 * ctx.etas)
    end

    _fld(fs, k) = hasproperty(fs, k) ? Float64(getproperty(fs, k)) : NaN
    R     = _fld(flow_state, :R)
    X     = _fld(flow_state, :X)
    psi   = _fld(flow_state, :psi)
    N1_p  = _fld(flow_state, :N1_p)
    K_til = _fld(flow_state, :K_til)

    return UsyextPoint(idx, estrain, t,
                       Wi, De, Y,
                       etaE1, etaE2, Tr1, Tr2,
                       tau.xx, tau.yy, tau.zz,
                       R, X, psi, N1_p, K_til,
                       status)
end

# ── Error norm ────────────────────────────────────────────────────────────────

function _err_norm(M0::Matrix{Float64}, M1::Matrix{Float64},
                   M2::Matrix{Float64}, tol::TolOpts)::Float64
    e = 0.0
    @inbounds for i in eachindex(M0)
        d  = abs(M2[i] - M1[i])
        sc = max(tol.atol + tol.rtol * max(abs(M0[i]), abs(M2[i])), tol.scale_floor)
        e  = max(e, d / sc)
    end
    return isfinite(e) ? e : Inf
end

# ── Step-size proposal ────────────────────────────────────────────────────────

function _propose_step(s::Float64, errn::Float64, tol::TolOpts)::Float64
    if !isfinite(errn) || errn <= 0
        return s * tol.grow_max
    end
    return s * tol.safety * errn^(-0.5)   # ETD2 is 2nd order
end

# ── Mode selection (with hysteresis) ─────────────────────────────────────────

function _decide_mode(ind::Symbol, Wi::Float64, Wi_min::Float64)::Symbol
    ind === :strain && return :strain
    ind === :time   && return :time
    return Wi > Wi_min ? :strain : :time   # :auto, no hysteresis (initial only)
end

function _decide_mode_hyst(ind::Symbol, Wi::Float64, Wi_min::Float64,
                            current::Symbol)::Symbol
    ind === :strain && return :strain
    ind === :time   && return :time
    # :auto with hysteresis band [Wi_min, 2*Wi_min]
    Wi_high = 2.0 * Wi_min
    if Wi < 0.0
        return :time
    elseif current === :strain && Wi <= Wi_min
        return :time
    elseif current === :time   && Wi >= Wi_high
        return :strain
    else
        return current
    end
end

# ── Core ETDRK2 step: strain mode ────────────────────────────────────────────

"""
    _step_strain_pair(ctx, flow, estrain, t, M, h)
        -> (M1, M2, estrain_a, t_etd1, t_heun)

One ETD1/ETD2 pair stepping by Δε = h in Hencky strain.

Returns the ETD1 predictor M1 and ETD2 corrected M2,
plus the updated strain/time estimates.
"""
function _step_strain_pair(ctx::UsyextCtx, flow,
                            estrain::Float64, t::Float64,
                            M::Matrix{Float64}, h::Float64)
    Nm = ctx.Nm

    # Stage n
    fs_n, theta_n, sigma_n, _, _ = _eval_state(ctx, flow, t, estrain, M)
    Wi_n = fs_n.Wi
    De   = fs_n.De

    # Build L (Nm×3) and b (Nm×3)
    # L[p,α] = 2α_α - θ_p/Wi_n
    # b[p,α] = σ_p/Wi_n
    L = Matrix{Float64}(undef, Nm, 3)
    b = Matrix{Float64}(undef, Nm, 3)
    @inbounds for p in 1:Nm
        for α in 1:3
            L[p,α] = 2.0 * ctx.alphas[α] - theta_n[p] / Wi_n
            b[p,α] = sigma_n[p] / Wi_n
        end
    end

    z    = L .* h
    expz = exp.(z)
    phi1, phi2 = _phi1phi2(z)

    # ETD1 predictor (= stage a)
    M1        = expz .* M .+ h .* phi1 .* b
    estrain_a = estrain + h
    t_a       = t + h * De / Wi_n   # Euler time advance

    # Stage a
    fs_a, theta_a, sigma_a, _, _ = _eval_state(ctx, flow, t_a, estrain_a, M1)
    Wi_a = fs_a.Wi

    # Stage-a RHS = (2α - θ_a/Wi_a).M1 + σ_a/Wi_a
    rhs_a = Matrix{Float64}(undef, Nm, 3)
    @inbounds for p in 1:Nm
        for α in 1:3
            rhs_a[p,α] = (2.0*ctx.alphas[α] - theta_a[p]/Wi_a)*M1[p,α] + sigma_a[p]/Wi_a
        end
    end
    N_a = rhs_a .- L .* M1   # nonlinear remainder at stage a
    N_n = b                   # at stage n (= b, since L*M + b is exact linear part)

    # ETD2 corrected
    M2 = expz .* M .+ h .* (phi1 .* N_n .+ phi2 .* (N_a .- N_n))

    # Heun time update: dt/dε = De/Wi
    t_etd1 = t + h * (De / Wi_n)
    t_heun = t + 0.5 * h * (De/Wi_n + De/Wi_a)

    return M1, M2, estrain_a, t_etd1, t_heun
end

# ── Core ETDRK2 step: time mode ───────────────────────────────────────────────

"""
    _step_time_pair(ctx, flow, estrain, t, M, dt)
        -> (M1, M2, estrain_etd1, estrain_heun, t_new)

One ETD1/ETD2 pair stepping by Δt = dt in dimensionless time.
"""
function _step_time_pair(ctx::UsyextCtx, flow,
                          estrain::Float64, t::Float64,
                          M::Matrix{Float64}, dt::Float64)
    Nm = ctx.Nm

    # Stage n
    fs_n, theta_n, sigma_n, _, _ = _eval_state(ctx, flow, t, estrain, M)
    Wi_n = fs_n.Wi
    De   = fs_n.De

    # dM/dt: L[p,α] = 2α_α·(Wi/De) - θ_p/De,   b[p,α] = σ_p/De
    L = Matrix{Float64}(undef, Nm, 3)
    b = Matrix{Float64}(undef, Nm, 3)
    @inbounds for p in 1:Nm
        for α in 1:3
            L[p,α] = 2.0*ctx.alphas[α]*(Wi_n/De) - theta_n[p]/De
            b[p,α] = sigma_n[p] / De
        end
    end

    z    = L .* dt
    expz = exp.(z)
    phi1, phi2 = _phi1phi2(z)

    # ETD1 predictor (= stage a)
    M1         = expz .* M .+ dt .* phi1 .* b
    estrain_a  = estrain + dt * (Wi_n / De)   # Euler strain advance
    t_a        = t + dt

    # Stage a
    fs_a, theta_a, sigma_a, _, _ = _eval_state(ctx, flow, t_a, estrain_a, M1)
    Wi_a = fs_a.Wi

    rhs_a = Matrix{Float64}(undef, Nm, 3)
    @inbounds for p in 1:Nm
        for α in 1:3
            rhs_a[p,α] = (2.0*ctx.alphas[α]*(Wi_a/De) - theta_a[p]/De)*M1[p,α] + sigma_a[p]/De
        end
    end
    N_a = rhs_a .- L .* M1
    N_n = b

    # ETD2 corrected
    M2 = expz .* M .+ dt .* (phi1 .* N_n .+ phi2 .* (N_a .- N_n))

    # Heun strain update: dε/dt = Wi/De
    estrain_heun = estrain + 0.5 * dt * (Wi_n/De + Wi_a/De)

    return M1, M2, estrain_a, estrain_heun, t_a
end

# ── Adaptive wrappers ─────────────────────────────────────────────────────────

function _try_strain_adapt(ctx::UsyextCtx, flow,
                            estrain::Float64, t::Float64,
                            M::Matrix{Float64}, dstrain::Float64,
                            tol::TolOpts)
    local M1, M2, estrain_a, t1, t2
    try
        M1, M2, estrain_a, t1, t2 = _step_strain_pair(ctx, flow, estrain, t, M, dstrain)
    catch
        # treat any error as a rejected step (e.g. LUT out-of-domain)
        ds_new = max(tol.shrink_min * dstrain, tol.dstrain_min)
        return false, M, M, estrain+dstrain, t, t, ds_new, Inf
    end

    errn = _err_norm(M, M1, M2, tol)
    ok   = (errn <= 1.0)
    ds_p = _propose_step(dstrain, errn, tol)
    ds_new = ok ? min(ds_p, tol.grow_max * dstrain) :
                  max(ds_p, tol.shrink_min * dstrain)

    return ok, M1, M2, estrain_a, t1, t2, ds_new, errn
end

function _try_time_adapt(ctx::UsyextCtx, flow,
                          estrain::Float64, t::Float64,
                          M::Matrix{Float64}, dt::Float64,
                          tol::TolOpts)
    local M1, M2, estrain_a, estrain_h, t_new
    try
        M1, M2, estrain_a, estrain_h, t_new = _step_time_pair(ctx, flow, estrain, t, M, dt)
    catch
        dt_new = max(tol.shrink_min * dt, tol.dt_min)
        return false, M, M, estrain, estrain, t+dt, dt_new, Inf
    end

    errn   = _err_norm(M, M1, M2, tol)
    ok     = (errn <= 1.0)
    dt_p   = _propose_step(dt, errn, tol)
    dt_new = ok ? min(dt_p, tol.grow_max * dt) :
                  max(dt_p, tol.shrink_min * dt)

    return ok, M1, M2, estrain_a, estrain_h, t_new, dt_new, errn
end

# ── Main integrator ───────────────────────────────────────────────────────────

"""
    integrate_usyext(ctx, flow, plan, tol) -> Vector{UsyextPoint}

Run the adaptive ETDRK2 USYEXT integrator.

Arguments
---------
ctx  : UsyextCtx  — polymer model context
flow : ConstantWiFlow | CaberFlow — flow protocol
plan : IntegrationPlan — sampling schedule + controller seeds
tol  : TolOpts — adaptive controller parameters

Returns a vector of UsyextPoint, one per sample strain.
The last entry may have status=:terminated if a limit was hit early.
"""
function integrate_usyext(ctx  :: UsyextCtx,
                           flow,
                           plan :: IntegrationPlan,
                           tol  :: TolOpts;
                           M0   :: Union{Matrix{Float64}, Nothing} = nothing)::Vector{UsyextPoint}

    Nm  = ctx.Nm
    ind = plan.independent

    # ── Initial state ────────────────────────────────────────────────────────
    # Default = equilibrium coil  M(0) = (1/3) I.  A non-equilibrium M0 (e.g. an
    # affine pre-stretch) may be supplied via the keyword; it must be Nm×3.
    if M0 === nothing
        M = fill(1.0/3.0, Nm, 3)
    else
        size(M0) == (Nm, 3) || error("integrate_usyext: M0 must be $(Nm)×3, got $(size(M0))")
        M = copy(M0)
    end
    estrain = 0.0
    t       = 0.0

    sample_strains = plan.sample_strains
    nS             = length(sample_strains)
    traj           = Vector{UsyextPoint}(undef, nS)

    # Seed step sizes (adaptive controller updates these)
    dstrain_cur = plan.dstrain_init
    dt_cur      = plan.dt_init

    n_accept = 0;  n_reject = 0
    wall_t0  = time()   # per-cell wallclock budget (tol.max_wall_s)

    # ── Initial snapshot (k=1) ───────────────────────────────────────────────
    fs_0, _, _, fp_0, Y_0 = _eval_state(ctx, flow, t, estrain, M)
    traj[1] = _make_obs(ctx, fs_0, M, fp_0, Y_0, 1, estrain, t, :ok)

    mode_current = _decide_mode(ind, fs_0.Wi, plan.Wi_min_strain_independent)

    k = 2
    while k <= nS
        target = sample_strains[k]

        while estrain < target
            rem = target - estrain
            if rem <= tol.hit_eps
                estrain = target
                break
            end

            # ── Safety caps: attempts and wallclock ───────────────────────
            # Checked at the TOP of the loop so reject-dominated grinds are
            # covered too (a `continue` after a reject must not skip the
            # budget).  Stiff cells (the inertio-capillary arg≈0 arrest) bail
            # here with a partial trajectory instead of hanging the sweep;
            # the elastic-corner Wi minimum typically lies well before the
            # grind, so downstream Wi-feature extraction still works.
            attempts = n_accept + n_reject
            if attempts > tol.max_attempts ||
               (isfinite(tol.max_wall_s) && (attempts & 0xfff == 0) &&
                time() - wall_t0 > tol.max_wall_s)
                why = attempts > tol.max_attempts ? "max_attempts=$(tol.max_attempts)" :
                                                    "max_wall_s=$(tol.max_wall_s)"
                @warn "integrate_usyext: step budget exceeded ($why) at ε=$estrain " *
                      "after $attempts attempts — returning partial trajectory"
                fs_c, _, _, fp_c, Y_c = _eval_state(ctx, flow, t, estrain, M)
                traj[k] = _make_obs(ctx, fs_c, M, fp_c, Y_c, k, estrain, t, :terminated)
                return traj[1:k]
            end

            # Current state for mode decision
            fs_cur, _, _, _, _ = _eval_state(ctx, flow, t, estrain, M)
            mode_current = _decide_mode_hyst(ind, fs_cur.Wi,
                                             plan.Wi_min_strain_independent,
                                             mode_current)

            estrain_prev = estrain;  t_prev = t;  M_prev = copy(M)

            if mode_current === :strain
                # ── strain-mode adaptive step ─────────────────────────────
                ds = min(dstrain_cur, rem)
                if rem >= tol.dstrain_min
                    ds = max(ds, tol.dstrain_min)
                else
                    ds = rem
                end

                ok, M1, M2, estrain_a, _, t_heun, ds_new, _ =
                    _try_strain_adapt(ctx, flow, estrain, t, M, ds, tol)

                if !ok
                    n_reject += 1
                    dstrain_cur = ds_new
                    isfinite(dstrain_cur) && dstrain_cur > 0 ||
                        error("integrate_usyext: non-positive dstrain after reject at ε=$estrain")
                    continue
                end

                n_accept    += 1
                estrain      = estrain_a
                t            = t_heun
                M            = M2
                dstrain_cur  = clamp(ds_new, tol.dstrain_min, tol.dstrain_max)

            else
                # ── time-mode adaptive step ───────────────────────────────
                fs_tm = fs_cur
                epsdot0 = max(fs_tm.Wi / fs_tm.De, tol.epsdot_floor)
                dt_hit  = rem / epsdot0

                dt = min(dt_cur, dt_hit)
                if dt_hit >= tol.dt_min
                    dt = max(dt, tol.dt_min)
                else
                    dt = dt_hit
                end

                ok, M1, M2, _, estrain_h, t_new, dt_new, _ =
                    _try_time_adapt(ctx, flow, estrain, t, M, dt, tol)

                if !ok
                    n_reject += 1
                    dt_cur = dt_new
                    isfinite(dt_cur) && dt_cur > 0 ||
                        error("integrate_usyext: non-positive dt after reject at ε=$estrain")
                    continue
                end

                n_accept  += 1
                estrain    = estrain_h
                t          = t_new
                M          = M2
                dt_cur     = clamp(dt_new, tol.dt_min, tol.dt_max)
            end

            # ── Overshoot correction (both modes) ─────────────────────────
            if estrain > target + tol.overshoot_eps
                w       = clamp((target - estrain_prev) / (estrain - estrain_prev), 0.0, 1.0)
                estrain = target
                t       = t_prev + w * (t - t_prev)
                M       = M_prev .+ w .* (M .- M_prev)
            end

            # ── Termination checks ────────────────────────────────────────
            if !all(isfinite, M)
                fs_t, _, _, fp_t, Y_t = _eval_state(ctx, flow, t, estrain, M)
                pt = _make_obs(ctx, fs_t, M, fp_t, Y_t, k, estrain, t, :terminated)
                traj[k] = pt
                return traj[1:k]
            end

            # Check Y
            fs_t, _, _, fp_t, Y_t = _eval_state(ctx, flow, t, estrain, M)
            if !isfinite(Y_t) || Y_t > ctx.Ymax
                pt = _make_obs(ctx, fs_t, M, fp_t, Y_t, k, estrain, t, :terminated)
                traj[k] = pt
                return traj[1:k]
            end

        end  # inner while

        # Record sample
        fs_k, _, _, fp_k, Y_k = _eval_state(ctx, flow, t, target, M)
        traj[k] = _make_obs(ctx, fs_k, M, fp_k, Y_k, k, target, t, :ok)
        k += 1
    end

    return traj
end

# ── CSV writer ────────────────────────────────────────────────────────────────

"""
    write_summary_csv(path, traj)

Write USYEXT trajectory to CSV in the standard reference format.

Columns: idx,estrain,t,tWi_ref,Wi,De,E2,etaE1,etaE2,Tr1,Tr2,tau_xx,tau_yy,tau_zz,status
"""
function write_summary_csv(path::AbstractString, traj::Vector{UsyextPoint})
    open(path, "w") do io
        println(io, "idx,estrain,t,tWi_ref,Wi,De,E2,etaE1,etaE2,Tr1,Tr2,tau_xx,tau_yy,tau_zz,status")
        for pt in traj
            @printf(io, "%d,%g,%g,NaN,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%s\n",
                    pt.idx, pt.estrain, pt.t,
                    pt.Wi, pt.De, pt.E2,
                    pt.etaE1, pt.etaE2,
                    pt.Tr1, pt.Tr2,
                    pt.tau_xx, pt.tau_yy, pt.tau_zz,
                    string(pt.status))
        end
    end
end

"""
    write_caber_diag_csv(path, traj)

Write CaBER-specific diagnostics to CSV, mirroring MATLAB's caber_diag.csv.

Columns: idx,estrain,t,Wi,De,R,X,N1_p,K_til,capstress,polystress,viscstress

Only meaningful for CaBER runs; for constant-Wi runs R/X/N1_p/K_til will be NaN.

Stress decomposition (from 0-D CaBER balance  3·Wi = capstress + polystress):
  capstress  = ((2X-1)/R)·De
  polystress = K̃·N1_p
  viscstress = capstress + polystress  (= 3·Wi from the balance)
"""
function write_caber_diag_csv(path::AbstractString, traj::Vector{UsyextPoint})
    open(path, "w") do io
        println(io, "idx,estrain,t,Wi,De,R,X,N1_p,K_til,capstress,polystress,viscstress")
        for pt in traj
            R     = pt.R
            X     = pt.X
            N1_p  = pt.N1_p
            K_til = pt.K_til
            De    = pt.De
            Wi    = pt.Wi

            if isnan(R) || isnan(X) || isnan(K_til)
                # Non-CaBER: write NaN for derived quantities
                capstress  = NaN
                polystress = NaN
                viscstress = NaN
            else
                capstress  = ((2.0*X - 1.0) / R) * De
                polystress = K_til * N1_p
                viscstress = capstress + polystress   # = 3·Wi by construction
            end

            @printf(io, "%d,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g\n",
                    pt.idx, pt.estrain, pt.t,
                    Wi, De, R, X, N1_p, K_til,
                    capstress, polystress, viscstress)
        end
    end
end

end # module UsyextIntegrator
