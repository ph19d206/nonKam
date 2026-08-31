"""
    find_converged_m(t_array, qfi_array;
                     transient_fraction=0.0, min_points=8,
                     min_abs_slope=0.05, tol=0.05) -> NamedTuple

Extract the *intermediate* power-law exponent `m` from positive data obeying
`f(t) ∝ t^m`.  The fit is performed in log--log space.  Rather than fitting
the end of the trace, this function finds the longest contiguous interval
whose local log--log slope is both stable and non-zero.  This excludes the
short-time transient and the artificial late-time saturation caused by a
finite Fock-space cutoff.

# Keyword arguments
- `transient_fraction`: Initial fraction of local slopes to ignore. The
  default is zero, so the stability test determines where a continuously
  changing early-time slope has settled. Set this above zero only to exclude
  a known initial interval.
- `min_points`: Minimum number of original data points in a candidate fit.
- `min_abs_slope`: Slopes smaller than this are treated as the saturation
  plateau.  Lower it if the physical exponent is close to zero.
- `tol`: Maximum relative standard deviation of the local slopes in the
  selected interval.

# Returns
A NamedTuple with `m_converged`, `m_std`, `t_converged`, `t_mid`, and
`m_local`.  `m_converged` is retained for backwards compatibility.
"""
function find_converged_m(
    t_array::AbstractVector{<:Real},
    qfi_array::AbstractVector{<:Real};
    transient_fraction::Real = 0.0,
    min_points::Integer = 8,
    min_abs_slope::Real = 0.05,
    tol::Real = 0.05,
)
    length(t_array) == length(qfi_array) ||
        throw(ArgumentError("t_array and qfi_array must have the same length."))
    0 <= transient_fraction < 1 ||
        throw(ArgumentError("transient_fraction must be in [0, 1)."))
    min_points >= 5 || throw(ArgumentError("min_points must be at least 5."))
    min_abs_slope >= 0 || throw(ArgumentError("min_abs_slope must be non-negative."))
    tol > 0 || throw(ArgumentError("tol must be positive."))

    valid = (t_array .> 0) .& (qfi_array .> 0)
    t = collect(float.(t_array[valid]))
    qfi = collect(float.(qfi_array[valid]))
    length(t) >= min_points ||
        error("Not enough positive data points to determine a power-law slope.")
    all(diff(t) .> 0) ||
        throw(ArgumentError("t_array must be strictly increasing after filtering."))

    log_t = log.(t)
    log_qfi = log.(qfi)
    n = length(t)
    m_local = [(log_qfi[i + 1] - log_qfi[i - 1]) /
               (log_t[i + 1] - log_t[i - 1]) for i in 2:n-1]
    t_mid = t[2:n-1]

    # A local-slope interval i:j corresponds to original data points i:j+2.
    min_local_points = min_points - 2
    first_allowed = max(1, floor(Int, transient_fraction * length(m_local)) + 1)
    last_allowed = length(m_local)
    best = nothing

    # Choose the longest stable, non-flat region.  For equal lengths, choose
    # the later one: it is farther from the short-time transient.
    for i in first_allowed:(last_allowed - min_local_points + 1)
        for j in (i + min_local_points - 1):last_allowed
            slopes = @view m_local[i:j]
            m_mean = mean(slopes)
            abs(m_mean) >= min_abs_slope || continue
            m_std = std(slopes)
            m_std <= tol * abs(m_mean) || continue

            candidate = (i=i, j=j, m_std=m_std)
            if isnothing(best) || (j - i > best.j - best.i) ||
               (j - i == best.j - best.i && i > best.i)
                best = candidate
            end
        end
    end

    isnothing(best) && error(
        "No stable non-flat power-law region found. Try increasing `tol`, " *
        "lowering `min_abs_slope`, or supplying more pre-saturation data."
    )

    fit_indices = best.i:(best.j + 2)
    x_fit = @view log_t[fit_indices]
    y_fit = @view log_qfi[fit_indices]
    A = hcat(x_fit, ones(length(x_fit)))
    m_fit, _ = A \ y_fit

    return (
        m_converged = m_fit,
        m_std = best.m_std,
        t_converged = t[fit_indices],
        t_mid = t_mid,
        m_local = m_local,
    )
end

"""
    find_power_law_regimes(t_array, values;
                           max_regimes=3, min_points=8,
                           slope_separation=1.0) -> NamedTuple

Split positive data into up to `max_regimes` contiguous power-law regions in
log--log space.  A boundary is retained only when the mean local slopes on
its two sides differ by at least `slope_separation`.  This is useful when a
trace contains an early-time scaling, an intermediate scaling, and a final
finite-basis saturation regime (slope close to zero).

Each entry in the returned `regimes` vector contains the fitted exponent `m`,
its local-slope spread `m_std`, the fitted `t_range`, and the corresponding
`value_range`.
"""
function find_power_law_regimes(
    t_array::AbstractVector{<:Real},
    values::AbstractVector{<:Real};
    max_regimes::Integer = 3,
    min_points::Integer = 8,
    slope_separation::Real = 1.0,
)
    length(t_array) == length(values) ||
        throw(ArgumentError("t_array and values must have the same length."))
    1 <= max_regimes <= 3 ||
        throw(ArgumentError("max_regimes must be between 1 and 3."))
    min_points >= 5 || throw(ArgumentError("min_points must be at least 5."))
    slope_separation >= 0 ||
        throw(ArgumentError("slope_separation must be non-negative."))

    valid = (t_array .> 0) .& (values .> 0)
    t = collect(float.(t_array[valid]))
    y = collect(float.(values[valid]))
    length(t) >= min_points ||
        error("Not enough positive data points to determine power-law regimes.")
    all(diff(t) .> 0) ||
        throw(ArgumentError("t_array must be strictly increasing after filtering."))

    log_t = log.(t)
    log_y = log.(y)
    m_local = [(log_y[i + 1] - log_y[i - 1]) /
               (log_t[i + 1] - log_t[i - 1]) for i in 2:length(t)-1]
    t_mid = t[2:end-1]
    n_local = length(m_local)
    min_local = min_points - 2
    n_local >= min_local || error("Not enough local slopes to split into regimes.")

    # Prefix sums give the within-region local-slope residual sum of squares
    # in O(1), so testing all two- and three-region partitions stays cheap.
    prefix = [0.0; cumsum(m_local)]
    prefix_sq = [0.0; cumsum(abs2.(m_local))]
    function segment_stats(first_idx, last_idx)
        count = last_idx - first_idx + 1
        total = prefix[last_idx + 1] - prefix[first_idx]
        total_sq = prefix_sq[last_idx + 1] - prefix_sq[first_idx]
        μ = total / count
        rss = max(0.0, total_sq - count * μ^2)
        return (mean=μ, rss=rss)
    end

    # Start with one segment and prefer the greatest number of well-separated
    # regimes.  Among equally sized models, choose the least variable split.
    best_ranges = [1:n_local]
    best_score = segment_stats(1, n_local).rss

    if max_regimes >= 2 && n_local >= 2 * min_local
        best_two = nothing
        for boundary in min_local:(n_local - min_local)
            left = segment_stats(1, boundary)
            right = segment_stats(boundary + 1, n_local)
            abs(left.mean - right.mean) >= slope_separation || continue
            score = left.rss + right.rss
            if isnothing(best_two) || score < best_two.score
                best_two = (ranges=[1:boundary, boundary + 1:n_local], score=score)
            end
        end
        if !isnothing(best_two)
            best_ranges = best_two.ranges
            best_score = best_two.score
        end
    end

    if max_regimes >= 3 && n_local >= 3 * min_local
        best_three = nothing
        for first_boundary in min_local:(n_local - 2 * min_local)
            left = segment_stats(1, first_boundary)
            for second_boundary in (first_boundary + min_local):(n_local - min_local)
                middle = segment_stats(first_boundary + 1, second_boundary)
                right = segment_stats(second_boundary + 1, n_local)
                abs(left.mean - middle.mean) >= slope_separation || continue
                abs(middle.mean - right.mean) >= slope_separation || continue
                score = left.rss + middle.rss + right.rss
                if isnothing(best_three) || score < best_three.score
                    best_three = (
                        ranges=[1:first_boundary,
                                first_boundary + 1:second_boundary,
                                second_boundary + 1:n_local],
                        score=score,
                    )
                end
            end
        end
        if !isnothing(best_three)
            best_ranges = best_three.ranges
            best_score = best_three.score
        end
    end

    regimes = map(best_ranges) do local_range
        # Local slope k is centred at data point k + 1.  These ranges are
        # disjoint, so every displayed fitted segment has an unambiguous time
        # interval even at a regime boundary.
        data_range = (first(local_range) + 1):(last(local_range) + 1)
        x_fit = @view log_t[data_range]
        y_fit = @view log_y[data_range]
        A = hcat(x_fit, ones(length(x_fit)))
        m_fit, log_prefactor = A \ y_fit
        local_values = @view m_local[local_range]
        (
            m=m_fit,
            m_std=std(local_values),
            t_range=t[data_range],
            value_range=y[data_range],
            prefactor=exp(log_prefactor),
        )
    end

    return (
        regimes=regimes,
        t_mid=t_mid,
        m_local=m_local,
        score=best_score,
    )
end
