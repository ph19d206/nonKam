# Plot_mean_energy_vs_time

# --------- Parameters ---------
# KHO parameters
R = 2
ω = 2*π/R
τ = 1 #π/2
K = 0.5
#-- 2nd set of parameters -- ωτ = 2π/R
# R = 5
# ν = 2/π
# τ = 1/ν
# ω = 2*π*ν/R
# ω = 4/R
# K = 1.5

# Simulation parameters
N = 1000          # Hilbert space dimension
t_max = 100      # Maximum time steps for energy calculation

# coherent state parameters
x0, p0 = -0.5903, -8.1437  # Initial phase space coordinates for the coherent state
# x0, p0 = 0.0121 + 3*π, 0.0121

# fitting parameters (Note: energy growth often follows different power laws than QFI!)
num_power = 2.0 # fitting power for number state (e.g., t^1 for normal diffusion)
cho_power = 1.8 # fitting power for coherent state.

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
# Assuming you named the function coherent_state in your module
α1 = coherent_state(x0, p0, ω, N) 

# ladder operators and Floquet unitary
a = diagm(1 => sqrt.(1:N-1))
adag = a'
U = KickedHO_floquet(N, ω, τ, K)

# Calculate Mean Energy over time for both states
# Using the Schrödinger picture function we created earlier
energy_n0 = mean_energy_with_time_for_U_a(U, a, adag, t_max, n0)
energy_α1 = mean_energy_with_time_for_U_a(U, a, adag, t_max, α1)

# Plotting
time_array = 1:t_max

fig, ax = plt.subplots(figsize=(6, 5))
ax.tick_params(axis="both", labelsize=18)

# -- plot data --
ax.plot(time_array, energy_α1, label="\$\\left|x+ip\\right>\$")  # Coherent state
ax.plot(time_array, energy_n0, label="\$\\left|0\\right>\$")

# -- plot fitted lines (optional) --
# You will likely need to adjust the prefactors (e.g., 800 and 0.7) for energy
ax.plot(time_array,  0.2 * time_array.^cho_power, "r--", label="\$t^{$cho_power}\$")  
ax.plot(time_array,  0.05 * time_array.^num_power, "k--", label="\$t^{$num_power}\$")  

# Add labels
ax.set_xlabel("\$t\$", fontsize=18)
# Changed y-axis label to represent mean energy <n>
ax.set_ylabel("\$\\langle \\hat{n} \\rangle\$", fontsize=18) 
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(false) # Adds a background grid
ax.legend()

# 3. Save
mkpath("plot")
# fig.savefig("plot/my_plot.pdf", bbox_inches="tight")
fig.savefig("plot/Plot_mean_energy_vs_time_two_state_singleR.pdf", bbox_inches="tight")

# 4. Clean up memory
plt.close(fig)