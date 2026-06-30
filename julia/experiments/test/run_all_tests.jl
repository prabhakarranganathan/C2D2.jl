"""
    run_all_tests.jl

Unified regression-test runner for the C2D2 Julia port.

Runs every test file in julia/experiments/test/ using the correct Julia project
for each suite, collects pass/fail/error counts, and prints a single summary table.

Usage (from repo root):
  julia --project=julia julia/experiments/test/run_all_tests.jl

Exit code: 0 if all suites pass, 1 if any suite fails or errors.
"""

using Dates
using Printf

const _REPO     = joinpath(@__DIR__, "..", "..", "..")
const _TEST_DIR = @__DIR__

# ── ANSI colours ──────────────────────────────────────────────────────────────

const _GRN  = "\e[32m"
const _RED  = "\e[31m"
const _YEL  = "\e[33m"
const _BLD  = "\e[1m"
const _RST  = "\e[0m"

_green(s)  = _GRN * s * _RST
_red(s)    = _RED * s * _RST
_yellow(s) = _YEL * s * _RST
_bold(s)   = _BLD * s * _RST

# ── Suite descriptors ─────────────────────────────────────────────────────────

struct Suite
    label   :: String          # display label
    file    :: String          # absolute path
    project :: String          # absolute path to Project.toml directory
    required :: Bool           # false → skip if project absent
end

const _JULIA    = joinpath(_REPO, "julia")

const SUITES = [
    Suite("kernel",       joinpath(_TEST_DIR, "test_kernel.jl"),       _JULIA,    true),
    Suite("evals",        joinpath(_TEST_DIR, "test_evals.jl"),         _JULIA,    true),
    Suite("styext",       joinpath(_TEST_DIR, "test_styext.jl"),        _JULIA,    true),
    Suite("usyext",       joinpath(_TEST_DIR, "test_usyext.jl"),        _JULIA,    true),
]

# ── Parse test output ──────────────────────────────────────────────────────────

"""
    _parse_counts(output) -> (pass, fail, err)

Extract aggregate pass/fail/error counts from Julia Test output.

Strategy:
1. On failure, Julia always prints:
       "N passed, M failed, K errored, 0 broken."
   Parse that line directly.
2. On full success (exit code 0), parse the "|  P  T  time" table data rows.
   Each top-level @testset produces one such row; when everything passes Julia
   does NOT show nested testset rows, so there is no double-counting.
   Table data rows always have a pipe `|` followed by two or more integers;
   the time column is identified by the last column (ends in [smh]).
"""
function _parse_counts(output::String)
    # ── Strategy 1: canonical failure summary line ────────────────────────────
    m = match(r"(\d+) passed,\s*(\d+) failed,\s*(\d+) errored", output)
    if m !== nothing
        return parse(Int, m.captures[1]), parse(Int, m.captures[2]),
               parse(Int, m.captures[3])
    end

    # ── Strategy 2: all-pass — parse "| P T time" data rows ──────────────────
    # Data rows: " label  |  N  N  <time>" where <time> contains letters.
    # Header rows: " Test Summary: | Pass  Total  Time" — contain non-digit text
    # after the pipe, so the regex (requiring \d+ first) won't match them.
    pass = 0
    for m in eachmatch(r"\|\s+(\d+)\s+(\d+)\b", output)
        # Both captured groups are integers (pass count, then total or fail)
        pass += parse(Int, m.captures[1])
    end
    return pass, 0, 0
end

# ── Runner ────────────────────────────────────────────────────────────────────

struct Result
    suite   :: Suite
    pass    :: Int
    fail    :: Int
    err     :: Int
    elapsed :: Float64
    status  :: Symbol          # :pass | :fail | :error | :skip
    output  :: String
end

function _run_suite(s::Suite)::Result
    if !s.required && !isfile(s.file)
        return Result(s, 0, 0, 0, 0.0, :skip, "")
    end

    julia_exe = joinpath(Sys.BINDIR, "julia")
    cmd = `$julia_exe --project=$(s.project) $(s.file)`

    t0  = time()
    out = IOBuffer()
    try
        proc = run(pipeline(cmd, stdout=out, stderr=out); wait=true)
        elapsed = time() - t0
        output  = String(take!(out))

        pass, fail, err = _parse_counts(output)

        if proc.exitcode != 0 || fail > 0 || err > 0
            return Result(s, pass, fail, err, elapsed, :fail, output)
        else
            return Result(s, pass, fail, err, elapsed, :pass, output)
        end
    catch e
        elapsed = time() - t0
        output  = String(take!(out)) * "\n" * sprint(showerror, e)
        return Result(s, 0, 0, 1, elapsed, :error, output)
    end
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    println()
    println(_bold("═══ C2D2 Julia port — full test suite ═══"))
    println("Started: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println()

    results = Result[]
    for s in SUITES
        @printf("  %-20s ... ", s.label)
        flush(stdout)
        r = _run_suite(s)
        push!(results, r)

        if r.status === :pass
            @printf("%s  (%d passed, %.1fs)\n",
                    _green("PASS"), r.pass, r.elapsed)
        elseif r.status === :skip
            println(_yellow("SKIP") * "  (project absent)")
        elseif r.status === :fail
            @printf("%s  (%d pass, %d fail, %d err, %.1fs)\n",
                    _red("FAIL"), r.pass, r.fail, r.err, r.elapsed)
        else
            @printf("%s  (%.1fs)\n", _red("ERROR"), r.elapsed)
        end
    end

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass  = count(r -> r.status === :pass,  results)
    n_fail  = count(r -> r.status === :fail,  results)
    n_err   = count(r -> r.status === :error, results)
    n_skip  = count(r -> r.status === :skip,  results)
    n_run   = length(results) - n_skip
    total_pass = sum(r.pass for r in results)
    total_fail = sum(r.fail for r in results)
    total_err  = sum(r.err  for r in results)

    println()
    println(_bold("─── Summary ─────────────────────────────"))
    @printf("  Suites:  %d run, %d pass, %d fail%s\n",
            n_run, n_pass, n_fail + n_err,
            n_skip > 0 ? ", $n_skip skipped" : "")
    @printf("  Tests:   %d pass, %d fail, %d error\n",
            total_pass, total_fail, total_err)

    # Print failure details
    for r in results
        if r.status in (:fail, :error)
            println()
            println(_red("━━━ FAILURE: $(r.suite.label) ━━━"))
            # Print last 40 lines of output (errors + test failures)
            lines = split(r.output, '\n')
            start = max(1, length(lines) - 40)
            foreach(l -> println("  ", l), lines[start:end])
        end
    end

    println()
    all_ok = (n_fail == 0) && (n_err == 0) && (total_fail == 0) && (total_err == 0)
    if all_ok
        println(_green("✓  All tests passed."))
    else
        println(_red("✗  Some tests FAILED."))
    end
    println()

    return all_ok ? 0 : 1
end

exit(main())
