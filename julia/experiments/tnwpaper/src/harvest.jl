# Included inside module TNW

function _sweep_dirs(root::AbstractString)
    isdir(root) || error("No such generic_features directory: $root")
    dirs = sort(filter(name -> isdir(joinpath(root, name)), readdir(root)))
    return [joinpath(root, d) for d in dirs]
end

function harvest_manifest(root::AbstractString=default_generic_outputs_dir())
    rows = NamedTuple[]

    for sweep_dir in _sweep_dirs(root)
        meta = parse_sweep_folder(sweep_dir)

        has_wi_vs_phi = isfile(joinpath(sweep_dir, "wi_features_vs_phi.csv"))
        has_wi_vs_de = isfile(joinpath(sweep_dir, "wi_features_vs_De.csv"))
        has_wie_vs_de = isfile(joinpath(sweep_dir, "Wi_e_vs_De.csv"))

        n_setting_dirs = 0
        n_run_dirs = 0
        n_summary = 0
        n_selected = 0

        for setting_name in sort(readdir(sweep_dir))
            setting_dir = joinpath(sweep_dir, setting_name)
            isdir(setting_dir) || continue
            startswith(setting_name, ".") && continue
            n_setting_dirs += 1
            for run_name in readdir(setting_dir)
                run_dir = joinpath(setting_dir, run_name)
                isdir(run_dir) || continue
                n_run_dirs += 1
                n_summary += isfile(joinpath(run_dir, "summary.csv")) ? 1 : 0
                n_selected += isfile(joinpath(run_dir, "wi_features_selected.csv")) ? 1 : 0
            end
        end

        push!(rows, merge(meta, (
            sweep_dir = sweep_dir,
            n_setting_dirs = n_setting_dirs,
            n_run_dirs = n_run_dirs,
            n_summary = n_summary,
            n_features_selected = n_selected,
            has_wi_features_vs_phi = has_wi_vs_phi,
            has_wi_features_vs_De = has_wi_vs_de,
            has_Wi_e_vs_De = has_wie_vs_de,
        )))
    end

    return DataFrame(rows)
end

function _attach_metadata!(df::DataFrame, meta::NamedTuple)
    for (k, v) in pairs(meta)
        df[!, Symbol(k)] = fill(v, nrow(df))
    end
    return df
end

function _read_csv_union(path::AbstractString)
    return DataFrame(CSV.File(path; silencewarnings=true, missingstring=["", "NA", "NaN"]))
end

function _resolved_phi(sweep_meta::NamedTuple, setting_meta::NamedTuple)
    if hasproperty(setting_meta, :setting_var) && setting_meta.setting_var == "phi"
        return setting_meta.setting_value
    elseif hasproperty(sweep_meta, :phi) && !ismissing(sweep_meta.phi)
        return sweep_meta.phi
    else
        return missing
    end
end
function _iter_run_artifacts(root::AbstractString)
    items = NamedTuple[]

    for sweep_dir in _sweep_dirs(root)
        sweep_meta = parse_sweep_folder(sweep_dir)

        for setting_name in sort(readdir(sweep_dir))
            setting_dir = joinpath(sweep_dir, setting_name)
            isdir(setting_dir) || continue
            startswith(setting_name, ".") && continue

            setting_meta = parse_setting_folder(setting_dir)

            for run_name in sort(readdir(setting_dir))
                run_dir = joinpath(setting_dir, run_name)
                isdir(run_dir) || continue

                run_meta = parse_run_folder(run_dir)
                summary_csv = joinpath(run_dir, "summary.csv")
                selected_csv = joinpath(run_dir, "wi_features_selected.csv")

                phi = _resolved_phi(sweep_meta, setting_meta)

                push!(items, merge(
                    sweep_meta,
                    setting_meta,
                    run_meta,
                    (
                        phi = phi,
                        setting_dir = setting_dir,
                        run_dir = run_dir,
                        summary_csv = isfile(summary_csv) ? summary_csv : missing,
                        features_selected_csv = isfile(selected_csv) ? selected_csv : missing,
                    )
                ))
            end
        end
    end

    return items
end

function harvest_curves(root::AbstractString=default_generic_outputs_dir())
    parts = DataFrame[]

    for item in _iter_run_artifacts(root)
        ismissing(item.summary_csv) && continue
        df = _read_csv_union(item.summary_csv)
        _attach_metadata!(df, item)
        push!(parts, df)
    end

    isempty(parts) && return DataFrame()
    return reduce((a, b) -> vcat(a, b; cols=:union), parts)
end

function harvest_features_selected(root::AbstractString=default_generic_outputs_dir())
    parts = DataFrame[]

    for item in _iter_run_artifacts(root)
        ismissing(item.features_selected_csv) && continue
        df = _read_csv_union(item.features_selected_csv)
        _attach_metadata!(df, item)
        push!(parts, df)
    end

    isempty(parts) && return DataFrame()
    return reduce((a, b) -> vcat(a, b; cols=:union), parts)
end

function _iter_sweep_feature_artifacts(root::AbstractString)
    items = NamedTuple[]
    wanted = [
        (filename = "wi_features_vs_phi.csv", kind = "wi_features_vs_phi"),
        (filename = "wi_features_vs_De.csv", kind = "wi_features_vs_De"),
        (filename = "Wi_e_vs_De.csv", kind = "Wi_e_vs_De"),
    ]

    for sweep_dir in _sweep_dirs(root)
        sweep_meta = parse_sweep_folder(sweep_dir)
        for w in wanted
            path = joinpath(sweep_dir, w.filename)
            if isfile(path)
                push!(items, merge(sweep_meta, (
                    sweep_dir = sweep_dir,
                    sweep_feature_kind = w.kind,
                    sweep_feature_csv = path,
                )))
            end
        end
    end

    return items
end

function harvest_sweep_features(root::AbstractString=default_generic_outputs_dir())
    parts = DataFrame[]

    for item in _iter_sweep_feature_artifacts(root)
        df = _read_csv_union(item.sweep_feature_csv)
        _attach_metadata!(df, item)
        push!(parts, df)
    end

    isempty(parts) && return DataFrame()
    return reduce((a, b) -> vcat(a, b; cols=:union), parts)
end

"""
    harvest_all_generic_features(; root=..., outdir=...)

Build and write the four canonical harvest tables.
Returns a NamedTuple with the DataFrames.
"""
function harvest_all_generic_features(; root::AbstractString=default_generic_outputs_dir(), outdir::AbstractString=default_harvested_dir())
    ensure_dir(outdir)

    manifest = harvest_manifest(root)
    curves = harvest_curves(root)
    features_selected = harvest_features_selected(root)
    sweep_features = harvest_sweep_features(root)

    CSV.write(joinpath(outdir, "manifest.csv"), manifest)
    CSV.write(joinpath(outdir, "curves.csv"), curves)
    CSV.write(joinpath(outdir, "features_selected.csv"), features_selected)
    CSV.write(joinpath(outdir, "sweep_features.csv"), sweep_features)

    return (
        manifest = manifest,
        curves = curves,
        features_selected = features_selected,
        sweep_features = sweep_features,
        outdir = outdir,
    )
end
