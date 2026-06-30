# Included inside module TNW

function _token_to_float(tok)::Float64
    s = String(tok)
    s = replace(s, 'p' => '.')
    return parse(Float64, s)
end

function _try_token_to_float(tok)
    try
        return _token_to_float(tok)
    catch
        return missing
    end
end

function _flow_family_from_kind(kind::AbstractString)
    k = lowercase(String(kind))
    if occursin("styext", k)
        return "styext"
    elseif occursin("caber", k)
        return "caber"
    else
        return "unknown"
    end
end

function _drag_family_from_token(tok::AbstractString)
    t = lowercase(String(tok))
    if t == "constant"
        return "constant"
    elseif t == "c2d2"
        return "c2d2"
    else
        return t
    end
end

"""
    parse_sweep_folder(path)

Parse a top-level sweep directory name such as
`S03_caber_sweep_De_NK_3000_hsK_0p05_phi_1_drag_constant` or
`S02_styext_sweep_phi_NK_3000_hsK_0p05_drag_c2d2`.

Returns a NamedTuple with metadata extracted from the folder name.
"""
function parse_sweep_folder(path::AbstractString)
    folder = basename(path)

    m = match(r"^S(?<sid>\d+?)_(?<flow>styext|caber)_sweep_(?<sweepvar>phi|De)_NK_(?<NK>[^_]+)_hsK_(?<hK>[^_]+)(?:_phi_(?<phi>[^_]+))?_drag_(?<drag>[^_]+)$", folder)
    m === nothing && error("Could not parse sweep folder name: $folder")

    sid = parse(Int, m[:sid])
    flow = String(m[:flow])
    sweep_var = String(m[:sweepvar])
    NK = parse(Int, m[:NK])
    hK_star = _token_to_float(m[:hK])
    phi = (m[:phi] === nothing) ? missing : _token_to_float(m[:phi])
    drag = _drag_family_from_token(m[:drag])

    return (
        sweep_id = sid,
        sweep_folder = folder,
        flow_family = flow,
        sweep_var = sweep_var,
        NK = NK,
        hK_star = hK_star,
        phi = phi,
        drag_model = drag,
    )
end

"""
    parse_setting_folder(path)

Parse child setting directories such as `phi_0.1` or `De_10`.
Returns `(setting_var, setting_value)`.
"""
function parse_setting_folder(path::AbstractString)
    folder = basename(path)
    if startswith(folder, "phi_")
        return (setting_var = "phi", setting_value = _token_to_float(folder[5:end]))
    elseif startswith(folder, "De_")
        return (setting_var = "De", setting_value = _token_to_float(folder[4:end]))
    else
        error("Could not parse setting folder name: $folder")
    end
end

"""
    parse_run_folder(path)

Parse a leaf run folder such as
`fenepm_constant_Nm=1_De=0.03_2026-03-14_081847` or
`fenepm_c2d2_Nm=1_2026-03-14_081215`.
"""
function parse_run_folder(path::AbstractString)
    folder = basename(path)

    model_name = "unknown"
    drag_model = "unknown"
    Nm = missing
    De_run = missing

    toks = split(folder, '_')
    if length(toks) >= 2
        model_name = toks[1]
        drag_model = _drag_family_from_token(toks[2])
    end

    for tok in toks
        if startswith(tok, "Nm=")
            try
                Nm = parse(Int, split(tok, "=")[2])
            catch
                Nm = missing
            end
        elseif startswith(tok, "De=")
            try
                De_run = parse(Float64, split(tok, "=")[2])
            catch
                De_run = missing
            end
        end
    end

    return (
        run_folder = folder,
        run_id = folder,
        model_name = model_name,
        drag_model_run = drag_model,
        Nm = Nm,
        De_run = De_run,
    )
end

# -------------------------------------------------------------------------
# Repo-layout-aware path resolution
# -------------------------------------------------------------------------

function _first_existing_dir(cands::Vector{String})
    for p in cands
        if isdir(p)
            return normpath(p)
        end
    end
    return normpath(cands[1])
end

function _first_existing_file(cands::Vector{String})
    for p in cands
        if isfile(p)
            return normpath(p)
        end
    end
    return normpath(cands[1])
end

"""
    default_generic_outputs_dir()

Resolve the default generic production outputs directory.
Supports both:
  repo/experiments/tnwpaper/outputs/generic_features
and
  repo/matlab/experiments/tnwpaper/outputs/generic_features
"""
function default_generic_outputs_dir()
    cands = String[
        joinpath(repo_root(), "matlab", "experiments", "tnwpaper", "outputs", "generic_features"),
        joinpath(repo_root(), "experiments", "tnwpaper", "outputs", "generic_features"),
    ]
    return _first_existing_dir(cands)
end

"""
    default_parameter_sets_xlsx()

Resolve the default parameter-sets workbook path.
Supports both:
  repo/experiments/tnwpaper/parameter-sets.xlsx
and
  repo/matlab/experiments/tnwpaper/parameter-sets.xlsx
"""
function default_parameter_sets_xlsx()
    cands = String[
        joinpath(repo_root(), "matlab", "experiments", "tnwpaper", "parameter-sets.xlsx"),
        joinpath(repo_root(), "experiments", "tnwpaper", "parameter-sets.xlsx"),
    ]
    return _first_existing_file(cands)
end

default_harvested_dir() = joinpath(paper_root(), "outputs", "harvested", "generic_features")

ensure_dir(path::AbstractString) = (mkpath(path); path)