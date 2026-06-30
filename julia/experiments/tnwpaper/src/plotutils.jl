# Included inside module TNW

function gray_series(n::Integer; lo=0.15, hi=0.80)
    n <= 1 && return [:black]
    vals = collect(range(lo, hi; length=n))
    return [RGBf(v, v, v) for v in vals]
end

function pretty_num(x)
    if ismissing(x)
        return "missing"
    elseif abs(x - round(x)) < 1e-12
        return @sprintf("%d", Int(round(x)))
    elseif abs(x) >= 1 || x == 0
        return @sprintf("%.3g", x)
    else
        return @sprintf("%.3g", x)
    end
end

function filter_fenep_baseline(df::DataFrame)
    mask = trues(nrow(df))

    if :flow_family in names(df)
        mask .&= df.flow_family .== "caber"
    end
    if :drag_model in names(df)
        mask .&= df.drag_model .== "constant"
    elseif :drag_model_run in names(df)
        mask .&= df.drag_model_run .== "constant"
    end
    if :NK in names(df)
        mask .&= df.NK .== 3000
    end
    if :hK_star in names(df)
        mask .&= isapprox.(df.hK_star, 0.05; atol=1e-12)
    end

    return df[mask, :]
end

function maybe_log10!(ax; xlog=false, ylog=false)
    if xlog
        ax.xscale = log10
    end
    if ylog
        ax.yscale = log10
    end
    return ax
end

function save_paper_figure(fig, filename::AbstractString)
    paperfig_path = joinpath(julia_root(), "models", "utils", "paperfig.jl")
    include(paperfig_path)
    return PaperFig.paper_save(fig, filename)
end
