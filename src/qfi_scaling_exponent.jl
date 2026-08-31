# src/qfi_scaling_exponent.jl
"""
    qfi_scaling_exponent(N::Int, ω::Real, τ::Real, K::Real, ϵ::Real, tmax::Int, state; 
                         qfi_floor::Float64 = 1e-10,
                         tol::Float64 = 1e-3,
                         min_points::Int = 10) -> NamedTuple

Computes the QFI time series and extracts the asymptotic power-law exponent `m` 
by iteratively trimming early-time transient points until the slope stabilizes 
within `tol`.

# Keyword Arguments
- `qfi_floor::Float64 = 1e-10`: Minimum valid QFI value to prevent log(0) divergence.
- `tol::Float64 = 1e-3`: Convergence tolerance for slope changes (|m_k - m_{k+1}| < tol).
- `min_points::Int = 10`: Minimum number of data points remaining required for linear fit.

# Returns
A NamedTuple containing:
- `m`: Extracted converged scaling exponent.
- `m_err`: Combined uncertainty (fit error + plateau variance).
- `t_converged`: Time array of the final converged window.
- `qfi_converged`: QFI array of the final converged window.
- `slopes_history`: Array of slopes evaluated during the trimming process.
"""
function qfi_scaling_exponent(
    N::Int, 
    ω::Real, 
    τ::Real, 
    K::Real, 
    ϵ::Real, 
    tmax::Int, 
    state;
    qfi_floor::Float64 = 1e-10,
    tol::Float64 = 1e-3,
    min_points::Int = 10
)
    # 1. Compute time array t = 1, 2, ..., tmax and QFI time series
    t_raw = collect(1:tmax)
    qfi_raw = floquet_qfi(N, ω, τ, K, ϵ, tmax, state)

    # 2. Filter out invalid/unstable initial data (QFI below floor or non-positive)
    valid_mask = (qfi_raw .> qfi_floor) .& (t_raw .> 0)
    t_valid = t_raw[valid_mask]
    qfi_valid = qfi_raw[valid_mask]

    n_valid = length(t_valid)
    if n_valid < min_points
        error("Not enough valid QFI points above threshold ($qfi_floor). Found $n_valid, needed $min_points.")
    end

    # Convert to log-log space
    x_valid = log.(t_valid)
    y_valid = log.(qfi_valid)

    # 3. Iterative Trimming: Trim early transient points step-by-step
    slopes_history = Float64[]
    errors_history = Float64[]
    windows = []

    # Step size for trimming (trim ~1-2% of points per iteration)
    step_size = max(1, div(n_valid, 100))

    for start_idx in 1:step_size:(n_valid - min_points)
        x_sub = x_valid[start_idx:end]
        y_sub = y_valid[start_idx:end]
        n_pts = length(x_sub)

        # OLS Linear Regression in log-log space: y = m*x + c
        x_mean = mean(x_sub)
        y_mean = mean(y_sub)
        
        m_k = sum((x_sub .- x_mean) .* (y_sub .- y_mean)) / sum((x_sub .- x_mean).^2)
        
        # Calculate fit standard error
        y_pred = m_k .* x_sub .+ (y_mean - m_k * x_mean)
        residuals = y_sub .- y_pred
        s_err = sqrt(sum(residuals.^2) / (n_pts - 2))
        m_err_k = s_err / sqrt(sum((x_sub .- x_mean).^2))

        push!(slopes_history, m_k)
        push!(errors_history, m_err_k)
        push!(windows, start_idx)

        # 4. Check for convergence plateau (|m_k - m_{k-1}| < tol over consecutive steps)
        if length(slopes_history) >= 3
            recent_diff = abs(slopes_history[end] - slopes_history[end-1])
            prev_diff   = abs(slopes_history[end-1] - slopes_history[end-2])

            if recent_diff < tol && prev_diff < tol
                # Plateau reached! Stop early.
                break
            end
        end
    end

    # 5. Extract final converged values
    # Take the mean of the stable plateau (last 3 iterations)
    plateau_len = min(3, length(slopes_history))
    m_converged = mean(slopes_history[end-plateau_len+1:end])
    
    # Combined Error = Fit Error + Variance across the plateau
    fit_err = mean(errors_history[end-plateau_len+1:end])
    plateau_var = std(slopes_history[end-plateau_len+1:end])
    total_m_err = sqrt(fit_err^2 + plateau_var^2)

    # Final window indices
    final_start = windows[end]

    return (
        m = m_converged,
        m_err = total_m_err,
        t_converged = t_valid[final_start:end],
        qfi_converged = qfi_valid[final_start:end],
        slopes_history = slopes_history
    )
end