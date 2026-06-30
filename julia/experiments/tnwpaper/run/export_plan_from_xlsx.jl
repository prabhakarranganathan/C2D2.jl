"""
    export_plan_from_xlsx.jl  (Piece 1)

Read Plan_Exptl from parameter-sets.xlsx, filter to one dataset, and emit
a per-dataset plan TOML in the same format as MATLAB's
tnwpaper_export_plan_from_xlsx_dataset.m.

Usage (from repo root):
    julia --project=julia julia/experiments/tnwpaper/run/export_plan_from_xlsx.jl \\
        <dataset_name> <out_plan_toml> [hK_star_override]

    dataset_name     e.g. "gaillard", "calabrese", "gaillard_aq"
    out_plan_toml    path for the output TOML (will be created/overwritten)
    hK_star_override optional; if given, only rows with this hK_star are exported

Returns: the plan TOML path (also written to stdout).
"""

using XLSX
using DataFrames
using Printf

# ── paths ─────────────────────────────────────────────────────────────────────

const _REPO = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))

function _xlsx_path()
    # Julia-side copy is the source of truth for Julia runs (decoupled from the
    # legacy MATLAB tree). Falls back to the MATLAB copy if the Julia one is absent.
    p = joinpath(_REPO, "julia", "experiments", "tnwpaper", "parameter-sets.xlsx")
    if !isfile(p)
        p = joinpath(_REPO, "matlab", "experiments", "tnwpaper", "parameter-sets.xlsx")
    end
    isfile(p) || error("parameter-sets.xlsx not found in julia/ or matlab/ tnwpaper dirs")
    return p
end

function _templates_base()
    # template paths in the plan TOML are relative to matlab/experiments/
    return joinpath(_REPO, "matlab", "experiments")
end

# ── helpers ────────────────────────────────────────────────────────────────────

function _canonicalize(s)
    s = lowercase(strip(string(s)))
    s = replace(s, r"\s+" => "_")
    s = replace(s, r"[^a-z0-9_]" => "")
    return s
end

function _num2token(x::Real)
    s = repr(round(Float64(x); sigdigits=12))
    s = replace(s, "." => "p")
    s = replace(s, "-" => "m")
    return s
end

function _normalize_enabled(x)
    x isa Bool  && return x
    x isa Number && return x != 0
    s = lowercase(strip(string(x)))
    return s in ("1", "true", "yes", "y", "on")
end

function _to_float_or_nothing(x)
    x === missing && return nothing
    try
        v = Float64(x)
        isfinite(v) ? v : nothing
    catch
        nothing
    end
end

# Optional per-row Float from a (possibly absent/blank) column.
function _rowf(row, sym::Symbol)
    hasproperty(row, sym) || return nothing
    v = getproperty(row, sym)
    v === missing && return nothing
    return _to_float_or_nothing(v)
end

# Try several canonicalized column-name spellings (e.g. "U_R" -> :u_r and
# "UR" -> :ur both work for the inertial modulus), returning the first hit.
function _rowf_any(row, syms::Symbol...)
    for s in syms
        v = _rowf(row, s)
        v === nothing || return v
    end
    return nothing
end

# ── TOML value serializer ──────────────────────────────────────────────────────

function _toml_val(v)
    v isa AbstractString && return "\"$(string(v))\""
    v isa Bool            && return v ? "true" : "false"
    v isa Integer         && return string(Int(v))
    if v isa AbstractFloat
        # always use full repr to avoid rounding surprises
        return repr(Float64(v))
    end
    if v isa AbstractVector
        parts = join([_toml_val(x) for x in v], ", ")
        return "[$parts]"
    end
    return repr(v)
end

# ── De grid parser ─────────────────────────────────────────────────────────────
# CaBER De cells in parameter-sets.xlsx are stored as text like
# "[0.01, 0.03, 0.1, ..., 300]". XLSX.jl hands these back as String, which the
# old code silently dropped to a hardcoded 9-point default — losing e.g. anna's
# De=300. Parse the string into the actual grid; fall back to the default only if
# it is genuinely empty/unparseable.
const _DE_DEFAULT = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

function _parse_de_string(s)
    s === nothing && return copy(_DE_DEFAULT)
    str = strip(string(s))
    isempty(str) && return copy(_DE_DEFAULT)
    body = strip(str, ['[', ']', ' '])
    vals = Float64[]
    for tok in split(body, [',', ';'])
        t = strip(tok)
        isempty(t) && continue
        v = tryparse(Float64, t)
        v === nothing || push!(vals, v)
    end
    return isempty(vals) ? copy(_DE_DEFAULT) : vals
end

# ── phi parser (scalar or bracketed list) ──────────────────────────────────────
# Plan_Exptl rows carry a single phi per sample; Plan_Generic caber rows may carry
# a list like "[0.01, 0.1, 0.5, 1]". Always return a Vector{Float64}; a scalar
# becomes a one-element vector so the Plan_Exptl path is unchanged.
function _parse_phi(x)
    x isa Number         && return [Float64(x)]
    x isa AbstractVector && return Float64.(x)
    str  = strip(string(x))
    body = strip(str, ['[', ']', ' '])
    vals = Float64[]
    for tok in split(body, [',', ';'])
        t = strip(tok)
        isempty(t) && continue
        v = tryparse(Float64, t)
        v === nothing || push!(vals, v)
    end
    isempty(vals) && error("could not parse phi value(s): $(repr(x))")
    return vals
end

# ── folder name builder ────────────────────────────────────────────────────────

function _sweep_folder(sweep_id, flow_type, NK, hK_star, phi, drag_model)
    hk_tok  = _num2token(hK_star)
    nk_int  = Int(round(NK))
    drag    = string(drag_model)

    if flow_type == "styext"
        return @sprintf("S%02d_styext_sweep_phi_NK_%d_hsK_%s_drag_%s",
                        sweep_id, nk_int, hk_tok, drag)
    else
        phi_tok = _num2token(phi)
        tag     = flow_type == "caber_inertial" ? "caber_inertial_sweep_De" :
                                                   "caber_sweep_De"
        return @sprintf("S%02d_%s_NK_%d_hsK_%s_phi_%s_drag_%s",
                        sweep_id, tag, nk_int, hk_tok, phi_tok, drag)
    end
end

# ── plan TOML writer ───────────────────────────────────────────────────────────

function _write_plan_toml(io::IO, out_root, dataset, rows)
    println(io, "[production]")
    println(io, "out_root = \"$(out_root)\"")
    println(io)
    println(io, "[templates]")
    println(io, "styext = \"experiments/tnwpaper/inputs/tnwpaper_styext_sweep_phi.toml\"")
    println(io, "caber  = \"experiments/tnwpaper/inputs/tnwpaper_caber_sweep_de.toml\"")

    for row in eachrow(rows)
        flow_type  = lowercase(strip(string(row.flow_type)))
        sweep_id   = Int(row.sweep_id)
        NK         = Float64(row.nk)
        hK_star    = Float64(row.hk_star)
        fe_model   = string(row.fe_model)
        drag_model = string(row.drag_model)

        if flow_type == "styext"
            # styext sweeps over phi internally via concentration.phi_values.
            # The phi column may be a scalar (Plan_Exptl, single sample) or a
            # bracketed list (Plan_Generic, e.g. a c2d2 styext curve whose shape
            # is concentration-dependent and so needs the full phi ladder).
            # Emit a scalar for the single-phi case (unchanged Plan_Exptl output)
            # and a list when several phi are given.
            phi_vals = _parse_phi(row.phi)
            phi_out  = length(phi_vals) == 1 ? phi_vals[1] : phi_vals
            _emit_sweep_block(io, sweep_id, flow_type, NK, hK_star, phi_out,
                              fe_model, drag_model, Float64[], nothing, nothing)
        else
            # caber / caber_inertial: the phi column may be a scalar or a list;
            # emit one sweep block per phi (each gets its own phi-tagged folder).
            phi_vals = _parse_phi(row.phi)
            De_raw   = row.de
            De_vals  = De_raw isa AbstractVector ? Float64.(De_raw) :
                       De_raw isa Number          ? [Float64(De_raw)] :
                       _parse_de_string(De_raw)
            # Oh is the primary inertial input (2026-06-12 pivot); blank ⇒
            # 0.005 (Gaillard aqueous), matching the code default.
            Oh  = flow_type == "caber_inertial" ? _rowf_any(row, :oh)        : nothing
            X_R = flow_type == "caber_inertial" ? _rowf_any(row, :x_r, :xr) : nothing
            if flow_type == "caber_inertial" && Oh === nothing
                Oh = 0.005
            end
            for phi in phi_vals
                _emit_sweep_block(io, sweep_id, flow_type, NK, hK_star, phi,
                                  fe_model, drag_model, De_vals, Oh, X_R)
            end
        end
    end
end

# Emit a single [[sweeps]] block. For caber_inertial, Oh must be non-nothing.
function _emit_sweep_block(io::IO, sweep_id, flow_type, NK, hK_star, phi,
                           fe_model, drag_model, De_vals, Oh, X_R)
    folder   = _sweep_folder(sweep_id, flow_type, NK, hK_star, phi, drag_model)
    kind     = flow_type == "styext" ? "styext_phi" : "caber_De"
    template = flow_type == "styext" ? "styext" : "caber"

    println(io)
    println(io, "[[sweeps]]")
    println(io, "kind     = \"$(kind)\"")
    println(io, "folder   = \"$(folder)\"")
    println(io, "template = \"$(template)\"")
    println(io, "set = [")

    println(io, "  [\"polymer.NK\",      $(_toml_val(Int(round(NK))))]," )
    println(io, "  [\"polymer.hK_star\", $(_toml_val(hK_star))],")
    println(io, "  [\"model.fe_model\",  $(_toml_val(fe_model))],")
    println(io, "  [\"model.drag_model\",$(_toml_val(drag_model))],")

    if flow_type == "styext"
        println(io, "  [\"concentration.phi_values\", $(_toml_val(phi))],")
    else
        println(io, "  [\"concentration.phi\",       $(_toml_val(phi))],")
        # For caber_inertial the De column is interpreted as De_R.
        println(io, "  [\"flow.De_values\",          $(_toml_val(Float64.(De_vals)))]," )

        if flow_type == "caber_inertial"
            # Inertio-capillary regime: Oh is the primary input; the device
            # locus U_R = Oh·U_1/De_R is formed inside eval_flow per cell.
            Oh !== nothing ||
                error("caber_inertial row (sweep_id=$(sweep_id)) has no Oh (bug: default not applied)")
            println(io, "  [\"flow.kind\",               \"caber_inertial\"],")
            println(io, "  [\"flow.Oh\",                 $(_toml_val(Oh))],")
            X_R === nothing ||
                println(io, "  [\"flow.X_R\",                $(_toml_val(X_R))],")
        end
    end

    println(io, "]")
end

# ── main export function ───────────────────────────────────────────────────────

"""
    export_plan_from_xlsx(dataset_name, out_plan_toml; hK_star=nothing)

Filter Plan_Exptl for `dataset_name` and write a plan TOML to `out_plan_toml`.
If `hK_star` is given (Float64), only rows with that hK_star value are included.
Returns the output TOML path.
"""
function export_plan_from_xlsx(dataset_name::AbstractString, out_plan_toml::AbstractString;
                                hK_star::Union{Float64,Nothing}=nothing,
                                sheet::AbstractString="Plan_Exptl")
    xlsx_path = _xlsx_path()
    ds = lowercase(strip(string(dataset_name)))

    raw = DataFrame(XLSX.readtable(xlsx_path, sheet))

    # canonicalize column names
    rename!(raw, Dict(n => Symbol(_canonicalize(string(n))) for n in names(raw))...)

    # resolve enabled
    enabled_col = "enabled" in names(raw) ? raw.enabled : trues(nrow(raw))
    enabled     = [_normalize_enabled(x) for x in enabled_col]

    # resolve folder column
    folder_col = "folder" in names(raw) ? string.(raw.folder) : fill("", nrow(raw))

    # filter to dataset
    keep = [enabled[i] && lowercase(strip(folder_col[i])) == ds for i in 1:nrow(raw)]
    T = raw[keep, :]
    nrow(T) > 0 || error("No enabled rows for dataset '$ds' in Plan_Exptl")

    # optionally further filter to one hK_star
    if hK_star !== nothing
        hk_cn = "hk_star" in names(T) ? "hk_star" : "hK_star"
        keep2 = [abs(Float64(T[i, hk_cn]) - hK_star) < 1e-10 for i in 1:nrow(T)]
        T = T[keep2, :]
        nrow(T) > 0 || error("No rows for dataset '$ds' with hK_star=$hK_star")
    end

    # get hK_star from first row (all must be the same per dataset)
    hk_col_name = "hk_star" in names(T) ? "hk_star" : "hK_star"
    hk_vals = unique([Float64(v) for v in T[:, hk_col_name] if !ismissing(v)])
    hK = isempty(hk_vals) ? 0.05 : hk_vals[1]

    # build out_root from xlsx or default
    out_root_base = if "out_root" in names(T)
        v = strip(string(first(T.out_root)))
        isempty(v) ? "julia/experiments/tnwpaper/outputs/exptl_comparison/$ds" : v
    else
        "julia/experiments/tnwpaper/outputs/exptl_comparison/$ds"
    end
    out_root_base = rstrip(out_root_base, '/')

    # Normalise a bare experiment-relative base (e.g. Plan_Generic's
    # "tnwpaper/outputs/generic_behaviour") to a repo-root-relative path under
    # julia/experiments/, since make_prod_sweeps joins out_root with the repo
    # root. Plan_Exptl paths already start with "julia/" so this is a no-op
    # for them.
    if startswith(out_root_base, "tnwpaper/")
        out_root_base = "julia/experiments/" * out_root_base
    end

    # build full out_root: base/dataset/hsK_<token>
    # The xlsx out_root base may or may not already include the dataset name.
    # Normalise: if the base does NOT end with the dataset name, append it.
    hk_tok   = _num2token(hK)
    trial_tag = "hsK_$(hk_tok)"
    base_norm = rstrip(out_root_base, '/')
    out_root  = if endswith(base_norm, ds)
        "$(base_norm)/$(trial_tag)"
    else
        "$(base_norm)/$(ds)/$(trial_tag)"
    end

    # ensure parent dir exists
    mkpath(dirname(abspath(out_plan_toml)))

    open(out_plan_toml, "w") do io
        _write_plan_toml(io, out_root, ds, T)
    end

    println("[TNW] Exported plan:")
    println("      dataset   = $ds")
    println("      rows      = $(nrow(T))")
    println("      hK_star   = $hK")
    println("      out_root  = $out_root")
    println("      plan TOML = $out_plan_toml")
    return out_plan_toml
end

# ── CLI entry ──────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 ||
        error("Usage: export_plan_from_xlsx.jl <dataset> <out_plan_toml> [hK_star] [sheet]")
    ds       = ARGS[1]
    out_toml = ARGS[2]
    # Remaining args (any order): a numeric token is hK_star, a non-numeric
    # token is the sheet name (default Plan_Exptl). This lets Plan_Generic
    # datasets (e.g. generic_inertial_oh0p01) be exported without an hK filter.
    hk    = nothing
    sheet = "Plan_Exptl"
    for a in ARGS[3:end]
        isempty(strip(a)) && continue
        if occursin(r"^[0-9.eE+\-]+$", a)
            hk = parse(Float64, a)
        else
            sheet = a
        end
    end
    export_plan_from_xlsx(ds, out_toml; hK_star=hk, sheet=sheet)
end
