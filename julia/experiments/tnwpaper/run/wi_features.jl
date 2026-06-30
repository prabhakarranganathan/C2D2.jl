"""
    wi_features.jl  (Piece 3)

Wi-feature detection, TnW batch selection, and sweep-level aggregation.

Ports MATLAB's:
  - extract_wi_features_from_summary.m  (candidate detection using y_feature_extractor)
  - select_wi_features_tnw_batch.m      (batch selector)
  - Sweep aggregation logic in tnwpaper_styext_sweep_phi.m / tnwpaper_caber_sweep_de.m

Public API:
    extract_wi_features(summary_csv; x_mode="E2") -> (candidates_df, selected_df)
    select_tnw_batch(candidates_df)               -> selected_df (with selection_reason)
    write_feature_csvs(outdir, candidates_df, selected_df)
    aggregate_styext(feature_rows)                -> wi_features_vs_phi_df
    aggregate_caber(feature_rows)                 -> (wi_features_vs_De_df, Wi_e_vs_De_df)
"""

using CSV
using DataFrames
using Statistics

# ─────────────────────────────────────────────────────────────────────────────
# Smoothing helpers
# ─────────────────────────────────────────────────────────────────────────────

function _movmean(x::Vector{Float64}, w::Int)
    n = length(x)
    w = max(1, w)
    h = div(w, 2)
    out = similar(x)
    for i in 1:n
        a = max(1, i - h)
        b = min(n, i + h)
        v = x[a:b]
        v = filter(isfinite, v)
        out[i] = isempty(v) ? NaN : mean(v)
    end
    return out
end

function _movmedian(x::Vector{Float64}, w::Int)
    n = length(x)
    w = max(1, w)
    h = div(w, 2)
    out = similar(x)
    for i in 1:n
        a = max(1, i - h)
        b = min(n, i + h)
        v = sort(filter(isfinite, x[a:b]))
        out[i] = isempty(v) ? NaN : v[ceil(Int, length(v) / 2)]
    end
    return out
end

function _smooth(y::Vector{Float64}, w::Int)
    # movmedian then movmean (mirrors MATLAB's toolbox-free fallback)
    w = max(3, isodd(w) ? w : w + 1)
    return _movmean(_movmedian(y, w), w)
end

function _gradient(y::Vector{Float64}, x::Vector{Float64})
    n = length(y)
    d = similar(y)
    n >= 2 || return fill(0.0, n)
    d[1] = (y[2] - y[1]) / (x[2] - x[1] + eps())
    for i in 2:(n-1)
        dx = x[i+1] - x[i-1]
        d[i] = (y[i+1] - y[i-1]) / (abs(dx) < eps() ? eps() : dx)
    end
    d[n] = (y[n] - y[n-1]) / (x[n] - x[n-1] + eps())
    return d
end

# ─────────────────────────────────────────────────────────────────────────────
# Extrema / inflexion detection
# ─────────────────────────────────────────────────────────────────────────────

function _pctl(v, p)
    v2 = filter(isfinite, v)
    isempty(v2) && return 0.0
    sort!(v2)
    n = length(v2)
    n == 1 && return v2[1]
    r = 1 + (n - 1) * (p / 100)
    lo = floor(Int, r); hi = ceil(Int, r)
    lo == hi ? v2[lo] : v2[lo] + (r - lo) * (v2[hi] - v2[lo])
end

function _fill_nans_bidir(s::Vector{Float64})
    n = length(s)
    out = copy(s)
    last = NaN
    for i in 1:n
        if isfinite(out[i]); last = out[i]
        elseif isfinite(last); out[i] = last; end
    end
    last = NaN
    for i in n:-1:1
        if isfinite(out[i]); last = out[i]
        elseif isfinite(last); out[i] = last; end
    end
    replace!(x -> isnan(x) ? 0.0 : x, out)
    return out
end

function _enforce_sep(idx::Vector{Int}, weight::Vector{Float64}, sep::Int)
    isempty(idx) && return Int[]
    ord = sortperm(weight)           # keep smallest weight first (smallest |slope| near extremum)
    keep = Int[]
    for k in ord
        j = idx[k]
        if all(abs.(j .- keep) .>= sep)
            push!(keep, j)
        end
    end
    return sort(keep)
end

"""
    signchange_extrema(d1, mask_mid, min_sep, tol_factor) -> (min_idx, max_idx)

Find extrema by sign change of first derivative.
"""
function _signchange_extrema(d1::Vector{Float64}, mask_mid::BitVector, min_sep::Int, tol_factor::Float64)
    ref = _pctl(abs.(d1[mask_mid]), 95)
    tol = tol_factor * max(ref, eps())

    s = sign.(d1)
    s[abs.(d1) .< tol] .= NaN
    s = _fill_nans_bidir(s)

    ds = diff(s)
    max_idx = findall(ds .< 0) .+ 1     # + -> -
    min_idx = findall(ds .> 0) .+ 1     # - -> +

    max_idx = _enforce_sep(max_idx, abs.(d1[max_idx]), min_sep)
    min_idx = _enforce_sep(min_idx, abs.(d1[min_idx]), min_sep)

    return min_idx, max_idx
end

"""
    curvature_inflexions(d2, mask_mid, sep) -> infl_idx

Find inflexion points by zero crossings of second derivative.
"""
function _curvature_inflexions(d2::Vector{Float64}, mask_mid::BitVector, sep::Int)
    s2 = sign.(d2)
    replace!(x -> isnan(x) || !isfinite(x) ? 0.0 : x, s2)

    z2 = findall(i -> s2[i] * s2[i+1] < 0, 1:(length(s2)-1))
    isempty(z2) && return Int[]

    # cluster nearby sign changes
    groups = Vector{Vector{Int}}()
    g = [z2[1]]
    for k in 2:length(z2)
        if z2[k] - z2[k-1] <= sep
            push!(g, z2[k])
        else
            push!(groups, g)
            g = [z2[k]]
        end
    end
    push!(groups, g)

    n = length(d2)
    infl = Int[]
    for gs in groups
        candj = unique(vcat(gs, gs .+ 1))
        candj = filter(j -> 1 <= j <= n && mask_mid[j], candj)
        isempty(candj) && continue
        _, ii = findmin(abs.(d2[candj]))
        push!(infl, candj[ii])
    end
    return sort(unique(infl))
end

# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

function _clamp01(x)
    isfinite(x) || return 0.0
    clamp(x, 0.0, 1.0)
end

function _local_prom(yT::Vector{Float64}, j::Int, w::Int, kind::Symbol)
    n = length(yT)
    a = max(1, j - w); b = min(n, j + w)
    yy = yT[a:b]
    yc = yT[j]
    p = kind == :min ? maximum(yy) - yc : yc - minimum(yy)
    isfinite(p) ? max(0.0, p) : 0.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Core feature extractor
# ─────────────────────────────────────────────────────────────────────────────

struct _Cand
    id         :: Int
    idx        :: Int          # index into original (unsorted) arrays
    j          :: Int          # index into sorted arrays
    type       :: String       # "min" | "max" | "inflexion"
    x          :: Float64
    y          :: Float64      # Wi value
    score      :: Float64
    d1         :: Float64      # derivative at feature (NaN for inflexions here)
    d2         :: Float64      # second derivative (NaN for extrema)
end

"""
    y_feature_extract(x, y; kw...) -> Vector{_Cand}

Generic feature extractor: finds min/max/inflexion candidates in y(x).
Returns candidates sorted by x.

Keyword args:
  endpoint_guard   = 2     (ignore first/last N pts)
  smooth_win_ext   = 7     (smoothing window for extrema)
  smooth_win_inf   = 13    (smoothing window for inflexions)
  min_sep          = 5
  min_prom         = 0.03
  tol_factor       = 1e-6
  max_candidates   = 12
  do_log_transform = true  (auto log10 if all y>0 and span>5)
"""
function y_feature_extract(x::Vector{Float64}, y::Vector{Float64};
                            endpoint_guard::Int    = 2,
                            smooth_win_ext::Int    = 7,
                            smooth_win_inf::Int    = 13,
                            min_sep::Int           = 5,
                            min_prom::Float64      = 0.03,
                            tol_factor::Float64    = 1e-6,
                            max_candidates::Int    = 12,
                            do_log_transform::Bool = true)

    length(x) == length(y) || error("x and y must have the same length")
    valid = isfinite.(x) .& isfinite.(y)
    x0  = x[valid]; y0 = y[valid]; idx0 = findall(valid)
    length(x0) >= 5 || error("Need at least 5 finite points")

    # sort by x
    ord  = sortperm(x0)
    xS   = x0[ord]; yS = y0[ord]; idxS = idx0[ord]
    n    = length(xS)

    # endpoint mask
    guard = max(0, endpoint_guard)
    mask_mid = trues(n)
    mask_mid[1:min(guard, n)]             .= false
    mask_mid[max(1, n-guard+1):n]         .= false

    # log transform?
    yT = copy(yS)
    if do_log_transform && all(yS .> 0) && maximum(yS) / minimum(yS) >= 5
        yT = log10.(yS)
    end

    # smoothings
    yT_ext = _smooth(yT, smooth_win_ext)
    yT_inf = _smooth(yT, smooth_win_inf)

    # derivatives in INDEX space (uniform j = 1..n), NOT physical xS space.
    #
    # Why: xS (e.g. E2) can span 3+ orders of magnitude with non-uniform spacing.
    # The very first x-steps (E2 changes by ~4e-6 while log-Wi changes by ~0.03)
    # produce d1 values O(1000s), which blow up the 95th-pctl reference used by
    # _signchange_extrema → tol ≈ 0.006 → all d1 near the peak/valley (~0.001)
    # get NaN'd → no sign change found → no min/max candidates.
    #
    # Index-space d1 has uniform spacing = 1 everywhere, so the reference stays
    # proportional to the typical shape change per step, and the tol is safe.
    # The physical x values (xS) are still stored/reported in candidates.
    idx_x  = Float64.(1:n)
    d1_ext = _gradient(yT_ext, idx_x)
    d1_inf = _gradient(yT_inf, idx_x)
    d2_inf = _gradient(d1_inf, idx_x)

    # ── extrema (sign-change mode, matches MATLAB "sign" / "auto" fallback) ──
    min_idx, max_idx = _signchange_extrema(d1_ext, mask_mid, min_sep, tol_factor)
    min_idx = filter(j -> mask_mid[j], min_idx)
    max_idx = filter(j -> mask_mid[j], max_idx)

    # ── inflexions (curvature mode) ──
    infl_idx = _curvature_inflexions(d2_inf, mask_mid, min_sep)
    infl_idx = filter(j -> mask_mid[j], infl_idx)

    # ── scoring ──
    prom_vals = Float64[]
    for j in min_idx; push!(prom_vals, _local_prom(yT_ext, j, 2*min_sep, :min)); end
    for j in max_idx; push!(prom_vals, _local_prom(yT_ext, j, 2*min_sep, :max)); end
    prom_scale = max(_pctl(prom_vals, 95), eps())

    curv_ext   = [j > 1 && j < n ? abs(yT_ext[j+1] - 2*yT_ext[j] + yT_ext[j-1]) : 0.0 for j in 1:n]
    curv_scale = max(_pctl(curv_ext[mask_mid], 95), eps())

    cands = _Cand[]
    cid   = 0

    for j in min_idx
        prom  = _local_prom(yT_ext, j, 2*min_sep, :min)
        curv  = j > 1 && j < n ? max(0.0, -(yT_ext[j+1] - 2*yT_ext[j] + yT_ext[j-1])) : 0.0
        score = 0.70 * _clamp01(prom / prom_scale) + 0.30 * _clamp01(curv / curv_scale)
        cid  += 1
        push!(cands, _Cand(cid, idxS[j], j, "min", xS[j], yS[j], score, d1_ext[j], NaN))
    end
    for j in max_idx
        prom  = _local_prom(yT_ext, j, 2*min_sep, :max)
        curv  = j > 1 && j < n ? max(0.0, -(yT_ext[j+1] - 2*yT_ext[j] + yT_ext[j-1])) : 0.0
        score = 0.70 * _clamp01(prom / prom_scale) + 0.30 * _clamp01(curv / curv_scale)
        cid  += 1
        push!(cands, _Cand(cid, idxS[j], j, "max", xS[j], yS[j], score, d1_ext[j], NaN))
    end

    slope_scale = max(_pctl(abs.(d1_inf[mask_mid]), 95), eps())
    d2_adj      = vcat([0.0], d2_inf, [0.0])  # for jump calc
    jump_scale  = max(_pctl(abs.(diff(d2_inf)), 95), eps())

    for j in infl_idx
        # skip if covered by an extremum at the same j
        any(c.j == j && c.type in ("min","max") for c in cands) && continue
        s_slope = _clamp01(abs(d1_inf[j]) / slope_scale)
        jj = clamp(j, 2, n-1)
        jump = abs(d2_inf[min(jj+1,n)] - d2_inf[max(jj-1,1)])
        s_jump = _clamp01(jump / max(jump_scale, eps()))
        score = 0.60 * s_slope + 0.40 * s_jump
        cid  += 1
        push!(cands, _Cand(cid, idxS[j], j, "inflexion", xS[j], yS[j], score, d1_inf[j], d2_inf[j]))
    end

    # cap by score
    if length(cands) > max_candidates
        sort!(cands, by=c -> -c.score)
        cands = cands[1:max_candidates]
        sort!(cands, by=c -> c.j)
        cands = [_Cand(k, c.idx, c.j, c.type, c.x, c.y, c.score, c.d1, c.d2)
                 for (k, c) in enumerate(cands)]
    end

    return cands
end

# ─────────────────────────────────────────────────────────────────────────────
# DataFrame builder for candidates
# ─────────────────────────────────────────────────────────────────────────────

"""
    cands_to_df(cands, T_summary) -> DataFrame

Join the candidate structs with the original summary.csv row data.
T_summary must be the full summary DataFrame (1-indexed rows).
"""
function cands_to_df(cands::Vector{_Cand}, T::DataFrame, feature_source::String="candidate")
    isempty(cands) && return DataFrame()
    rows = DataFrame[]
    for c in cands
        idx = c.idx
        (idx < 1 || idx > nrow(T)) && continue
        r = T[idx:idx, :]
        r.feature_source = [feature_source]
        r.feature_type   = [c.type]
        r.feature_id     = [c.id]
        r.feature_idx    = [idx]
        r.feature_j      = [c.j]
        r.feature_score  = [c.score]
        r.feature_x      = [c.x]
        r.feature_Wi     = [c.y]
        r.feature_d1     = [c.d1]
        r.feature_d2     = [c.d2]
        push!(rows, r)
    end
    isempty(rows) && return DataFrame()
    df = vcat(rows...)
    # move feature_* columns to front
    front = ["feature_source","feature_type","feature_id","feature_idx","feature_j",
             "feature_score","feature_x","feature_Wi","feature_d1","feature_d2"]
    present = filter(c -> c in names(df), front)
    rest    = filter(c -> c ∉ front, names(df))
    return df[:, vcat(present, rest)]
end

# ─────────────────────────────────────────────────────────────────────────────
# Public: extract_wi_features
# ─────────────────────────────────────────────────────────────────────────────

"""
    extract_wi_features(summary_csv; x_mode="E2") -> (candidates_df, T_summary)

Read summary.csv and detect Wi feature candidates.
x_mode: "E2" (styext default) | "estrain" / "strain" (usyext default)
"""
function extract_wi_features(summary_csv::AbstractString; x_mode::String="E2")
    isfile(summary_csv) || error("summary.csv not found: $summary_csv")
    T = DataFrame(CSV.File(summary_csv; silencewarnings=true))

    :Wi in propertynames(T) || error("No Wi column in $summary_csv")

    Wi = Float64.(T.Wi)

    x = if lowercase(x_mode) in ("e2",)
        :E2 in propertynames(T) || error("No E2 column in $summary_csv")
        Float64.(T.E2)
    elseif lowercase(x_mode) in ("estrain","strain")
        col = :estrain in propertynames(T) ? :estrain :
              :strain  in propertynames(T) ? :strain  : nothing
        col === nothing && error("No estrain/strain column in $summary_csv")
        Float64.(T[!, col])
    elseif lowercase(x_mode) == "t"
        :t in propertynames(T) || error("No t column in $summary_csv")
        Float64.(T.t)
    else
        error("Unknown x_mode: $x_mode")
    end

    cands = y_feature_extract(x, Wi)
    cands_df = cands_to_df(cands, T, "candidate")
    return cands_df, T
end

# ─────────────────────────────────────────────────────────────────────────────
# Public: select_tnw_batch
# ─────────────────────────────────────────────────────────────────────────────

"""
    select_tnw_batch(candidates_df) -> selected_df

TnW batch selector (mirrors select_wi_features_tnw_batch.m):
  - ignore terminals
  - if any maxima: keep one with largest feature_Wi
  - if any minima: keep one with smallest feature_Wi
  - if no min/max: keep best inflexion (smallest |d2|, then largest |d1|, then largest score)
"""
function select_tnw_batch(C::DataFrame)
    isempty(C) && return DataFrame()

    :feature_type in propertynames(C) || return DataFrame()
    :feature_Wi   in propertynames(C) || return DataFrame()

    # ignore terminals
    not_terminal = map(t -> t ∉ ("start", "end"), string.(C.feature_type))
    C2 = C[not_terminal, :]
    isempty(C2) && return DataFrame()

    rows = DataFrame[]

    mins = C2[string.(C2.feature_type) .== "min", :]
    maxs = C2[string.(C2.feature_type) .== "max", :]
    infl = C2[string.(C2.feature_type) .== "inflexion", :]

    if !isempty(mins)
        _, jmin = findmin(Float64.(mins.feature_Wi))
        r = mins[jmin:jmin, :]
        r.selection_reason = ["deepest_min"]
        push!(rows, r)
    end

    if !isempty(maxs)
        _, jmax = findmax(Float64.(maxs.feature_Wi))
        r = maxs[jmax:jmax, :]
        r.selection_reason = ["highest_max"]
        push!(rows, r)
    end

    if isempty(mins) && isempty(maxs) && !isempty(infl)
        ad2    = [isfinite(v) ? abs(Float64(v)) : Inf  for v in infl.feature_d2]
        ad1    = [isfinite(v) ? abs(Float64(v)) : -Inf for v in infl.feature_d1]
        sc     = [isfinite(v) ? Float64(v) : -Inf      for v in infl.feature_score]
        xv     = [isfinite(v) ? Float64(v) : Inf       for v in infl.feature_x]

        # lexicographic sort: smallest |d2|, then largest |d1|, then largest score, then smallest x
        idx_sort = sortperm(1:length(ad2), by=i -> (ad2[i], -ad1[i], -sc[i], xv[i]))
        r = infl[idx_sort[1]:idx_sort[1], :]
        r.selection_reason = ["fallback_inflexion"]
        push!(rows, r)
    end

    isempty(rows) && return DataFrame()
    out = vcat(rows...)
    :feature_x in propertynames(out) && sort!(out, :feature_x)
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Public: write_feature_csvs
# ─────────────────────────────────────────────────────────────────────────────

"""
    write_feature_csvs(outdir, candidates_df, selected_df)

Write wi_feature_candidates.csv and wi_features_selected.csv into outdir.
"""
function write_feature_csvs(outdir::AbstractString,
                             cands_df::DataFrame,
                             sel_df::DataFrame)
    !isempty(cands_df) && CSV.write(joinpath(outdir, "wi_feature_candidates.csv"), cands_df)
    !isempty(sel_df)   && CSV.write(joinpath(outdir, "wi_features_selected.csv"),  sel_df)
end

# ─────────────────────────────────────────────────────────────────────────────
# Public: aggregate helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    aggregate_styext(feature_rows) -> wi_features_vs_phi_df

feature_rows: vector of (sweep_var, sweep_value, run_id, outdir, selected_df) tuples.
"""
function aggregate_styext(feature_rows)
    parts = DataFrame[]
    for (var, val, run_id, outdir, sel) in feature_rows
        isempty(sel) && continue
        r = copy(sel)
        r.sweep_var   = fill("phi", nrow(r))
        r.sweep_value = fill(Float64(val), nrow(r))
        r.run_id      = fill(string(run_id), nrow(r))
        r.outdir      = fill(string(outdir), nrow(r))
        # move sweep_* to front
        front = ["sweep_var","sweep_value","run_id","outdir"]
        rest  = filter(c -> c ∉ front, names(r))
        push!(parts, r[:, vcat(front, rest)])
    end
    isempty(parts) ? DataFrame() : vcat(parts...; cols=:union)
end

"""
    aggregate_caber(feature_rows) -> (wi_features_vs_De_df, Wi_e_vs_De_df)
"""
function aggregate_caber(feature_rows)
    feat_parts = DataFrame[]
    wie_rows   = DataFrame[]

    for (var, de_val, run_id, outdir, sel) in feature_rows
        isempty(sel) && continue
        r = copy(sel)
        r.sweep_var   = fill("De", nrow(r))
        r.sweep_value = fill(Float64(de_val), nrow(r))
        r.run_id      = fill(string(run_id), nrow(r))
        r.outdir      = fill(string(outdir), nrow(r))
        front = ["sweep_var","sweep_value","run_id","outdir"]
        rest  = filter(c -> c ∉ front, names(r))
        push!(feat_parts, r[:, vcat(front, rest)])

        # Wi_e_vs_De: one row per De — the MIN/plateau feature (NOT the max).
        # Mirrors MATLAB tnwpaper_postprocess_wie_vs_de.m: prefer feature_type
        # == "min" (deepest_min); else selection_reason == "fallback_inflexion";
        # else emit nothing for this De (never report the highest_max). Among
        # eligible rows, keep the one with the highest feature_score.
        ok = :status in propertynames(r) ? (string.(r.status) .== "ok") : trues(nrow(r))
        is_min = ok .& (string.(r.feature_type) .== "min")
        is_fb  = (:selection_reason in propertynames(r)) ?
                 (ok .& (string.(r.selection_reason) .== "fallback_inflexion")) :
                 falses(nrow(r))
        mask = any(is_min) ? is_min : (any(is_fb) ? is_fb : falses(nrow(r)))
        if any(mask)
            cand = r[mask, :]
            j    = argmax(Float64.(cand.feature_score))
            push!(wie_rows, DataFrame(
                De           = [Float64(de_val)],
                Wi_e         = [Float64(cand.feature_Wi[j])],
                feature_score= [Float64(cand.feature_score[j])],
            ))
        end
        # else: no usable min/fallback feature for this De → skip (MATLAB warns + skips)
    end

    feat_df = isempty(feat_parts) ? DataFrame() : vcat(feat_parts...; cols=:union)
    wie_df  = isempty(wie_rows)   ? DataFrame() : vcat(wie_rows...)
    isempty(wie_df)  || sort!(wie_df,  :De)
    isempty(feat_df) || sort!(feat_df, :sweep_value)
    return feat_df, wie_df
end
