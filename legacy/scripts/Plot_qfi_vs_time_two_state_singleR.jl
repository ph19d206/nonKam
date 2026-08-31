# Plot_qfi_vs_time

# --------- Parameters ---------
# KHO parameters
R = 5
ω = 2/π
τ = 2*π/(ω*R)
K = 1.5
#-- 2nd set of parameters -- ωτ = 2π/R
# R = 5
# ν = 2/π
# τ = 1/ν
# ω = 2*π*ν/R
# ω = 4/R
# K = 1.5

# Simulation parameters
N = 600          # Hilbert space dimension
t_max = 100      # Maximum time steps for QFI calculation

# coherent state parameters
x0, p0 = -0.5903, -8.1437  # Initial phase space coordinates for the coherent state
# fitting parameters
num_power = 2.5 # fitting power for number state.
cho_power = 2 # fitting power for coherent state.

# --------- Main Execution ---------
# import packages
using LinearAlgebra
using nonKam
using PyCall

# import python packages for plotting
plt = pyimport("matplotlib.pyplot")
mpl = pyimport("matplotlib")
np = pyimport("numpy")


# Initial states (ground state)
# if state is change then change label in plot as well.
# -- number states --
n0 = zeros(ComplexF64, N)
n0[1] = 1.0 + 0im  # |0⟩ state

# -- coherent states --
α1 = coherent_state(x0, p0, ω, N)

# ladder operators and Floquet unitary
a = diagm(1 => sqrt.(1:N-1))
adag = a'
U = KickedHO_floquet(N, ω, τ, K)

# Calculate QFI over time for both states
qfi_n0 = qfi_with_time_provided_U_a(U, a, adag, ω, τ, K, t_max, n0)
qfi_α1 = qfi_with_time_provided_U_a(U, a, adag, ω, τ, K, t_max, α1)

# Plotting
time_array = 1:t_max

fig, ax = plt.subplots(figsize=(6, 5))
ax.tick_params(axis="both", labelsize=18)

# -- plot data --
ax.plot(time_array, qfi_α1, label="\$\\left|x+ip\\right>\$")  # Coherent state
ax.plot(time_array, qfi_n0, label="\$\\left|0\\right>\$")


# -- plot fitted lines (optional) --
ax.plot(time_array,  800*time_array.^cho_power, "r--", label="\$t^{$cho_power}\$")  # Example power-law fit
ax.plot(time_array,  0.7*time_array.^num_power, "k--", label="\$t^{$num_power}\$")  # Example power-law fit


# Add labels (optional but good practice)
# ax.set_title("Single Plot: ta^2")
ax.set_xlabel("\$t\$", fontsize=18)
ax.set_ylabel("\$I(\\left|\\psi\\right>;R)\$", fontsize=18)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(false) # Adds a background grid
ax.legend()

# 3. Save
mkpath("plot")
# fig.savefig("plot/my_plot.pdf", bbox_inches="tight")
fig.savefig("plot/Plot_qfi_vs_time_two_state_singleR.pdf", bbox_inches="tight")


# 4. Clean up memory
plt.close(fig)
