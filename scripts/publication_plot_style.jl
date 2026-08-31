"""Shared PythonPlot/Matplotlib settings for publication figures."""

const PLOT_COLORS = (
    blue = "#0072B2",
    vermillion = "#D55E00",
    green = "#009E73",
    purple = "#7B2CBF",
    black = "#222222",
)

"""
    apply_publication_style!(ax; scale=:loglog)

Apply consistent fonts, axes, and scaling. The scale may be :linear,
:loglog, :semilogx, or :semilogy.
"""
function apply_publication_style!(ax; scale::Symbol=:loglog)
    scale in (:linear, :loglog, :semilogx, :semilogy) ||
        throw(ArgumentError("scale must be :linear, :loglog, :semilogx, or :semilogy"))

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["STIXGeneral", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["legend.fontsize"] = 8
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.top"] = true
    mpl.rcParams["ytick.right"] = true

    scale in (:loglog, :semilogx) && ax.set_xscale("log")
    scale in (:loglog, :semilogy) && ax.set_yscale("log")

    ax.grid(false)
    ax.tick_params(which="both", width=0.8, length=3)
    for spine in ax.spines.values()
        spine.set_linewidth(0.8)
    end
end

"""Save a figure with the project's publication export settings."""
function save_publication_figure(fig, output_file::AbstractString)
    fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.03)
end
