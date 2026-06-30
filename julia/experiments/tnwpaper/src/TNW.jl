module TNW

using CSV
using DataFrames
using Statistics
using Printf
using XLSX
using CairoMakie

const _SRC_DIR = @__DIR__
const _TNW_DIR = normpath(joinpath(_SRC_DIR, ".."))
const _JULIA_DIR = normpath(joinpath(_SRC_DIR, "..", "..", ".."))
const _REPO_ROOT = normpath(joinpath(_SRC_DIR, "..", "..", "..", ".."))

repo_root() = _REPO_ROOT
julia_root() = _JULIA_DIR
paper_root() = _TNW_DIR

include("metadata.jl")
include("harvest.jl")
include("plotutils.jl")

export repo_root, julia_root, paper_root
export parse_sweep_folder, parse_setting_folder, parse_run_folder
export harvest_manifest, harvest_curves, harvest_features_selected, harvest_sweep_features
export harvest_all_generic_features, ensure_dir
export default_generic_outputs_dir, default_parameter_sets_xlsx, default_harvested_dir
export filter_fenep_baseline, gray_series, maybe_log10!, save_paper_figure, pretty_num

end # module
