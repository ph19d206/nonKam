#!/usr/bin/env julia
# Temporary test/demo for plot_curves (src/plot_helper.jl).
# Run from the repository root:
#     julia --project=. test/test_plot_curves.jl

using nonKam

const plt = nonKam.PythonPlot  # access to pyplot helpers

# ---------------------------------------------------------------
# Style 1: one shared x-axis with several y-curves
# ---------------------------------------------------------------
x  = 0:0.1:10
ys = [sin.(x), cos.(x), x .* 0.1]

fig, ax = plot_curves(x, ys,
                      xlabel="time", ylabel="amplitude",
                      labels=["sin", "cos", "linear"])

# You can keep extending the figure afterwards:
ax.plot(x, x.^0.5, "k--", label="added later")
ax.legend(fontsize=15)
fig.tight_layout()
fig.savefig("test_plot_curves_style1.pdf", dpi=150)
println("Saved test_plot_curves_style1.pdf")

# ---------------------------------------------------------------
# Style 2: matched (x, y) pairs
# ---------------------------------------------------------------
fig, ax = plot_curves([x, x], [sin.(x), cos.(x)],
                      xlabel="x", ylabel="y",
                      labels=["sin (pair)", "cos (pair)"])
fig.tight_layout()
fig.savefig("test_plot_curves_style2.pdf", dpi=150)
println("Saved test_plot_curves_style2.pdf")

# ---------------------------------------------------------------
# Style 3: two separately-created plots as side-by-side subplots
# of one single figure (pass an existing `ax` to plot_curves)
# ---------------------------------------------------------------
x2 = 0:0.1:10

# One figure with two subplots (side by side).  Change (1, 2) to (2, 1)
# to stack them vertically instead.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))

# First "separate" plot -> drawn into the left subplot.
plot_curves(x,  [sin.(x), cos.(x)],
            ax=axL, xlabel="time", title="left", legend=true)

# Second "separate" plot -> drawn into the right subplot.
plot_curves(x2, [sin.(2x2), cos.(2x2)],
            ax=axR, xlabel="time", title="right", legend=false)

fig.tight_layout()
fig.savefig("test_plot_curves_subplots.pdf", dpi=150)
println("Saved test_plot_curves_subplots.pdf")

# ---------------------------------------------------------------
# Defaults check: print the line colors used
# ---------------------------------------------------------------
fig, ax = plot_curves(x, [sin.(x), cos.(x), exp.(-0.3x)])
println("Colors used: ", [l.get_color() for l in ax.get_lines()])

plt.close("all")
println("All good.")
