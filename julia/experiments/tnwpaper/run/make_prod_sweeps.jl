"""
    make_prod_sweeps.jl  (Piece 2a)

Read a plan TOML and generate per-sweep folders, each containing a patched
input.toml and a copy of the plan.toml (mirrors tnwpaper_make_prod_sweeps.m).

Does NOT run any simulations — just prepares the folder tree.

Usage (from repo root):
    julia --project=julia julia/experiments/tnwpaper/run/make_prod_sweeps.jl <plan_toml>

Returns: list of (sweep_dir, input_toml, kind) tuples written to stdout.
"""

import TOML

# ── paths ─────────────────────────────────────────────────────────────────────

const _REPO_MPS = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))

function _matlab_exp_root()
    joinpath(_REPO_MPS, "matlab", "experiments")
end

function _julia_exp_root()
    joinpath(_REPO_MPS, "julia", "experiments")
end

# ── TOML text patching ────────────────────────────────────────────────────────
# Mirrors MATLAB's toml_patch(): find [section], find/insert key = value.

function _toml_val_str(v)
    v isa AbstractString && return "\"$(string(v))\""
    v isa Bool            && return v ? "true" : "false"
    v isa Integer         && return string(Int(v))
    v isa AbstractFloat   && return repr(Float64(v))
    if v isa AbstractVector
        parts = join([_toml_val_str(x) for x in v], ", ")
        return "[$parts]"
    end
    return repr(v)
end

function toml_patch!(lines::Vector{String}, keypath::String, value)
    parts      = split(keypath, ".")
    length(parts) >= 2 || error("toml_patch: expected table.key, got: $keypath")
    table_name = join(parts[1:end-1], ".")
    key_name   = parts[end]
    header     = "[$table_name]"
    newline    = "$key_name = $(_toml_val_str(value))"

    # find table header
    idx_header = findfirst(l -> strip(l) == header, lines)

    if idx_header === nothing
        # append table at end
        push!(lines, "")
        push!(lines, header)
        push!(lines, newline)
        return lines
    end

    # find end of this table (next [header] or EOF)
    idx_end = length(lines)
    for k in (idx_header+1):length(lines)
        s = strip(lines[k])
        if startswith(s, "[") && endswith(s, "]")
            idx_end = k - 1
            break
        end
    end

    # find existing key within table
    pat = Regex("^\\s*$(escape_string(key_name))\\s*=")
    idx_key = nothing
    for k in (idx_header+1):idx_end
        s = strip(lines[k])
        !startswith(s, "#") && !startswith(s, ";") && occursin(pat, lines[k]) && (idx_key = k; break)
    end

    if idx_key === nothing
        # insert just after header (skip leading blanks/comments)
        ins = idx_header + 1
        while ins <= length(lines)
            s = strip(lines[ins])
            (isempty(s) || startswith(s, "#") || startswith(s, ";")) ? ins += 1 : break
        end
        insert!(lines, ins, newline)
    else
        lines[idx_key] = newline
    end
    return lines
end

# ── apply set patches ─────────────────────────────────────────────────────────

function _apply_set!(lines::Vector{String}, sw::Dict)
    haskey(sw, "set") || return
    patches = sw["set"]
    patches isa AbstractVector || return
    for pair in patches
        pair isa AbstractVector && length(pair) == 2 || continue
        keypath = string(pair[1])
        val     = pair[2]
        toml_patch!(lines, keypath, val)
    end
end

# ── plan reader + sweep folder creator ───────────────────────────────────────

"""
    make_prod_sweeps(plan_toml) -> Vector{NamedTuple}

Read plan_toml, create sweep folders, write patched input.toml + plan.toml into each.
Returns a vector of (sweep_dir, input_toml, kind, folder) NamedTuples.
"""
function make_prod_sweeps(plan_toml::AbstractString)
    isfile(plan_toml) || error("Plan TOML not found: $plan_toml")
    plan = TOML.parsefile(plan_toml)

    prod  = get(plan, "production", Dict())
    out_root_rel = get(prod, "out_root", "")
    isempty(out_root_rel) && error("production.out_root missing in $plan_toml")

    # out_root is relative to repo root
    out_root_abs = joinpath(_REPO_MPS, out_root_rel)
    mkpath(out_root_abs)

    tmpl_sec = get(plan, "templates", Dict())

    sweeps = get(plan, "sweeps", nothing)
    (sweeps === nothing || isempty(sweeps)) && error("No [[sweeps]] entries in $plan_toml")
    sweeps isa AbstractVector || error("sweeps must be an array")

    matlab_exp = _matlab_exp_root()

    results = NamedTuple[]

    for (i, sw) in enumerate(sweeps)
        sw isa Dict || continue
        folder = get(sw, "folder", "")
        isempty(folder) && error("Sweep $i missing folder")
        kind   = lowercase(get(sw, "kind",     ""))
        tname  = lowercase(get(sw, "template", ""))

        isempty(tname) && error("Sweep $i ($folder) missing template")
        haskey(tmpl_sec, tname) || error("Template '$tname' not in [templates]")

        tmpl_path_rel = tmpl_sec[tname]
        # template paths: first try relative to matlab/experiments/, then matlab/
        tmpl_abs = joinpath(matlab_exp, tmpl_path_rel)
        if !isfile(tmpl_abs)
            tmpl_abs = joinpath(dirname(matlab_exp), tmpl_path_rel)
        end
        isfile(tmpl_abs) ||
            error("Template not found: $tmpl_path_rel")

        sweep_dir = joinpath(out_root_abs, folder)
        mkpath(sweep_dir)

        # read template, apply patches
        lines = readlines(tmpl_abs)
        toml_patch!(lines, "run.label",    folder)
        toml_patch!(lines, "run.out_root", out_root_rel)
        _apply_set!(lines, sw)

        input_toml = joinpath(sweep_dir, "input.toml")
        open(input_toml, "w") do io
            println(io, join(lines, "\n"))
        end

        cp(plan_toml, joinpath(sweep_dir, "plan.toml"); force=true)

        @printf("[%2d/%d] %s\n", i, length(sweeps), input_toml)
        push!(results, (sweep_dir=sweep_dir, input_toml=input_toml, kind=kind, folder=folder))
    end

    return results
end

# ── CLI entry ──────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || error("Usage: make_prod_sweeps.jl <plan_toml>")
    plan_toml = ARGS[1]
    results   = make_prod_sweeps(plan_toml)
    println("\n=== make_prod_sweeps: wrote $(length(results)) sweep folders ===")
    for r in results
        println("  $(r.sweep_dir)")
    end
end
