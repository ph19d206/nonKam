#!/usr/bin/env julia
using JLD2

ds = jldopen(ARGS[1], "r") do f
    Vector(f["datasets"])
end
for (i, d) in enumerate(ds)
    println("entry $i: sig=", d.signature, " R=", d.R, " n=", d.n,
            " x0=", d.x0, " p0=", d.p0)
    println("  number_fit.m  = ", d.number_fit.m)
    println("  coherent_fit.m = ", d.coherent_fit.m)
    println("  number_fit.t  = ", d.number_fit.t)
    println("  coherent_fit.t = ", d.coherent_fit.t)
    println("  energy_num head=", d.energy_number[1:3],
            " tail=", d.energy_number[end-2:end])
    println("  energy_coh head=", d.energy_coherent[1:3],
            " tail=", d.energy_coherent[end-2:end])
    println("  time head=", d.time[1:3], " tail=", d.time[end-2:end])
end