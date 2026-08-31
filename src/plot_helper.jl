# plot_helper.jl
# Convenience plotting wrapper around PythonPlot (matplotlib).
# Supports two calling styles:
#   * single x shared by several curves:  plot_curves(x, [y1, y2, ...])
#   * several (x, y) pairs:               plot_curves([x1, x2, ...], [y1, y2, ...])
# Returns the (fig, ax) pair so the caller can keep adding lines/text afterwards.

using PythonPlot
const plt = PythonPlot

# LaTeX-style rendering for axis labels and tick labels (mathtext, not usetex).
function _set_latex_style!(mpl)
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["DejaVu Serif", "STIXGeneral"]
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["axes.formatter.use_mathtext"] = true
end

# Paul Tol "bright" palette (colorblind-safe, publication-quality),
# used in e.g. "Scientific colour maps". Cycles after 7 curves.
const PLOT_COLORS = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494","#B3B3B3",];
# const PLOT_COLORS = ["#3B5B92",  "#D97757", "#4C956C", "#8C6BB1","#D4A72C", "#5AA6A6",  "#C06C84", "#4A4A4A" ]
# const PLOT_COLORS = ["#F5537E", "#E36C33", "#8FAE00", "#01B45A","#00B6C3", "#4F8FF2", "#C47DFF", "#FF9DA3"]
# const PLOT_COLORS = ["#4477B6", "#EE6676", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

"""
    fig, ax = plot_curves(x, ys; kwargs...)
    fig, ax = plot_curves(xs, ys; kwargs...)

Plot multiple curves with a shared default color palette and sensible fonts.

Styles:
  * `plot_curves(x, ys)`      where `x` is a vector and `ys` is a vector of vectors
                              (each `ys[i]` is one curve sharing the same `x`).
  * `plot_curves(xs, ys)`     where `xs` and `ys` are equal-length vectors of vectors
                              (each `(xs[i], ys[i])` is one curve).

Keyword arguments (all optional):
  * `xlabel`, `ylabel`  : axis labels (default "").
  * `title`             : plot title (default "", not shown if empty).
  * `labels`            : vector of legend labels. If not given, curves are
                          auto-labelled 1, 2, 3, ... when a legend is requested.
  * `legend`            : `true`/`false` (default `true`).
  * `label_fontsize`    : axis-label font size (default 18).
  * `tick_fontsize`     : tick font size (default 16).
  * `legend_fontsize`   : legend font size (default 15).
  * `figsize`           : figure size in inches, e.g. (8, 6) (default (8, 6)).
  * `return_ax`         : if `true`, return just `ax` (default `false`).
  * `linewidth`         : line width of every curve (default 3).
  * `ax`                : existing matplotlib axes to draw into. If given, no
                          new figure/axes is created; used to combine several
                          separately-created plots as subplots of one figure.

Extra keyword arguments (e.g. `marker`, `linestyle`) are passed straight to
every `ax.plot(...)` call.

The returned `(fig, ax)` let you add further curves/text afterwards; call
`fig`/`ax` display functions yourself (or `plt.show()`) once you are done.
"""
function plot_curves(x, ys;
                      xlabel::AbstractString="",
                      ylabel::AbstractString="",
                      title::AbstractString="",
                      labels=nothing,
                      legend::Bool=true,
                      label_fontsize::Integer=20,
                      tick_fontsize::Integer=20,
                      legend_fontsize::Integer=18,
                      figsize=(8, 6),
                      return_ax::Bool=false,
                      linewidth::Real=3,
                      ax=nothing,
                      kwargs...)

    xs = nothing
    # A "vector of vectors" is the marker for the matched-pairs style.
    isvecvec(v) = v isa AbstractVector && !isempty(v) &&
                  all(e -> e isa AbstractVector, v)
    if isvecvec(x)
        # style 2: xs and ys are matched vectors of vectors
        xs = [collect(xv) for xv in x]
        ys = [collect(yv) for yv in ys]
    else
        # style 1: shared x, ys is a vector of vectors
        xs = [collect(x) for _ in ys]
        ys = [collect(y) for y in ys]
    end

    # Sanity checks.
    if length(xs) != length(ys)
        error("plot_curves: got $(length(xs)) x-curves but $(length(ys)) y-curves.")
    end
    n = length(ys)
    for i in 1:n
        if length(xs[i]) != length(ys[i])
            error("plot_curves: curve $i has $(length(xs[i])) x-values but " *
                  "$(length(ys[i])) y-values.")
        end
    end

    _set_latex_style!(plt.pyimport("matplotlib"))
    if ax === nothing
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else
        # Draw into the caller-provided axes (e.g. a subplot of an existing
        # figure) instead of creating a new one, so several separately-created
        # plots can be combined into a single multi-panel figure.
        fig = ax.figure
    end

    if labels === nothing
        labels = [string(i) for i in 1:n]
    end

    for i in 1:n
        color = PLOT_COLORS[mod1(i, length(PLOT_COLORS))]
        ax.plot(xs[i], ys[i], color=color, label=labels[i],
                linewidth=linewidth; kwargs...)
    end

    # Axis styling.
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    if !isempty(title)
        ax.set_title(title, fontsize=label_fontsize)
    end
    if legend
        ax.legend(fontsize=legend_fontsize)
    end

    return return_ax ? ax : (fig, ax)
end

export plot_curves
