using Revise
using LinearAlgebra
using Statistics
using JLD2
using ITensors
using QuantumNaturalGradient
using QuantumNaturalfPEPS
using Distributed
using TimerOutputs
using MPI

mpi = true 
threaded = false
multiproc = false
shared_array = true
println("Threaded: ", threaded, ", Multiproc: ", multiproc)

# # Initialize 2 processes with 8 threads each, assuming you have a 16 core cpu
# nr_procs, nr_threads = 2,4
# addprocs(nr_procs, exeflags="-t $nr_threads")
# # atexit(() -> rmprocs(workers()))  # ensure cleanup on exit


# @everywhere begin
#     using QuantumNaturalfPEPS
#     using LinearAlgebra
#     BLAS.set_num_threads(1) # Each thread should only use one Blas thread
# end
# BLAS.set_num_threads(8) # For computing the double layers

# BLAS.set_num_threads(1)

# Simulation Parameters
params = Dict(
    :T              => ComplexF64,
    :Lx             => 4,
    :Ly             => 4,
    :bdim           => 4,
    :lr             => .01,
    :J1             => 1,
    :eigencut       => 1e-6,
    :contract_cutoff=> 1e-7,
    :sample_cutoff  => 1e-7,
    :contract_dim   => 200,
    :maxiter        =>1000,
    :α_init         => 2.0,
    :sample_nr      => 1500,
)
# Function to compute the structure factor S(q) for given q
function structure_factor(q::Tuple{Any,Any}, coords::Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}, corrmat::Matrix{Float64}, params::Dict)
    S_q = 0.0
    N = params[:Lx] * params[:Ly]
    for i in 1:params[:Lx], j in 1:params[:Ly]
        (xi, yi) = coords[(i,j)]
        idx = (i-1)*params[:Ly] + j
        # Si = expectations[idx]
        for k in 1:params[:Lx], l in 1:params[:Ly]
        #      if i == k && j == l
        #     continue
        # end

  
            (xk, yk) = coords[(k,l)]
            idx2 = (k-1)*params[:Ly] + l
            # Sk = expectations[idx2]
            r_ij = [xi - xk, yi - yk]
            # -- optional: apply PBC --
            # dx = xi - xk
            # dy = yi - yk
            # dx -= round(dx / params[:Lx]) * params[:Lx]
            # dy -= round(dy / params[:Ly]) * params[:Ly]
            # r_ij = [dx, dy]

            phase = exp(-im * dot(q, r_ij))
            S_q += corrmat[idx,idx2] * phase
        
        end
    end
    return real(S_q / N^2)

    
end


# Define the Lattice and PEPS
hilbert = siteinds("S=1/2", params[:Lx], params[:Ly])
peps = PEPS(params[:T], hilbert; bond_dim=params[:bdim], show_warning=true,
    contract_cutoff=params[:contract_cutoff], 
    contract_dim=params[:contract_dim], 
    sample_cutoff=params[:sample_cutoff],
)

# Multiply the spectrum of the PEPS by a power-law factor to make it contractible
QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps, params[:α_init])

# Construct the Hamiltonian
h_1d = OpSum()
ham_J1x = OpSum()
ham_J1y = OpSum()
ham_J1 = OpSum()


a = 1.0 #lattice constant
coords = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
coord_to_1d(i, j) = (i-1)*params[:Ly] + j


for i in 1:params[:Lx], j in 1:params[:Ly]
     y = (j-1) * sqrt(3)/2 
    x = (i-1) + 0.5*(j%2)
    coords[(i,j)] = (x,y)
    println("Coordinate of site ($i,$j) = ($(x),$(y))")
end


# Define the magnetic moment direction
ctheta = 1 # any angle in radians
m = [0, ctheta]         # magnetic moment or reference vector
println("Magnetic moment direction m = $m")


for i in 1:params[:Lx], j in 1:params[:Ly]
    (xi, yi) = coords[(i,j)]
    n = coord_to_1d(i, j)
    for k in 1:params[:Lx], l in 1:params[:Ly]
        # skip self-interaction
        if i == k && j == l
            continue
        end

        # avoid double-counting
        if (k < i) || (k == i && l < j)
            (xk, yk) = coords[(k,l)]

            # vector between sites
            rij = [xi - xk, yi - yk]

            # -- optional: apply PBC --
            # dx = xi - xk
            # dy = yi - yk
            # dx -= round(dx / params[:Lx]) * params[:Lx]
            # dy -= round(dy / params[:Ly]) * params[:Ly]
            # rij = [dx, dy]

            # cosine of angle between rij and m
            costheta = dot(rij, m) / (norm(rij) )
            # println("rij = $rij, costheta = $costheta")

            # interaction strength
            # if norm(rij)> 3.6
            #     println("Skipping interaction between ($i,$j) and ($k,$l) due to distance > 3.6")
            #     continue
            # end
            # interaction = params[:J1] * (1 - 3 * costheta^2) / (norm(rij)^3)
            
            interaction = -params[:J1]/(norm(rij)^3) # isotropic antiferromagnetic Heisenberg interaction
            ham_J1 .+= (interaction, "Sz", (i,j), "Sz", (k,l))
            ham_J1x .+= (interaction/2, "Sz", (i,j), "Sz", (k,l))
            ham_J1y .+= (interaction/2, "Sz", (i,j), "Sz", (k,l))

            println("Interaction between ($i,$j) and ($k,$l): $interaction")
            nk = coord_to_1d(k, l)
            # println("Adding interaction between ($i,$j) and ($k,$l) = $interaction")
            h_1d .+= (interaction, "Sz", n,"Sz" , nk)
            h_1d .+= (interaction/2, "Sx", n,"Sx" , nk) 
            h_1d .+= (interaction/2, "Sy", n,"Sy" , nk) 

            # println("Adding interaction between ($i,$j) and ($k,$l) = $interaction")
            # end
        end
    end
end
# Find the ground state using DMRG
sites = siteinds("S=1/2", params[:Lx]*params[:Ly])
H = MPO(h_1d, sites)
psi0 = randomMPS(sites, 12)
# psi0 = productMPS(sites, "Up")
energy, psi = dmrg(H, psi0; nsweeps=8,maxdim=256)
println("Ground state energy = $energy")
corr = real(correlation_matrix(psi, "Sz", "Sz"))
psi0 = nothing
psi = nothing
# println(corr)

println("Structure factor at q = (0,0) from DMRG = ", structure_factor((0, 0.0), coords, corr, params))
println("Structure factor at q = (pi,0) from DMRG = ", structure_factor((pi, 0), coords, corr, params))
println("Structure factor at q = (0,pi) from DMRG =", structure_factor((0, pi), coords, corr, params))
println("Structure factor at q = (pi,pi) from DMRG = ", structure_factor((pi, pi), coords, corr, params))
# println("Structure factor at q = (0,pi/3) from DMRG = ", structure_factor((0, pi/sqrt(3)), coords, corr, params))
# println("Structure factor at q = (0,2*pi/3) from DMRG = ", structure_factor((0, 2*pi/sqrt(3)), coords, corr, params))
# throw(InterruptException())

# Define a Callback Function, that is called after each iteration
best_theta = Ref{Any}(nothing)
best_energy = Ref(1e10) # used for rollback (store previous energy)
prev_energy = Ref(1e10) # used for rollback (store previous energy)
count = Ref(0)
count_best = Ref(0)
energy_arr =[]
function callback(;state, niter=1, kwargs...)
    kwargs = Dict(string(key) => v for (key, v) in kwargs)
    kwargs["params"] = params
    kwargs["solver"] = solver
    kwargs["integrator"] = integrator
    kwargs["peps"] = peps
    kwargs["timer"] = timer

    push!(energy_arr,state.energy)
    energy_change = state.energy - prev_energy[]

    # if energy_change < 0# Big improvement
    # #     # Keep LR the same or increase it slightly
    #     state.integrator.lr = min(1, 1.05 * state.integrator.lr)
    #     println("Energy decrease, increasing learning rate to ", state.integrator.lr)
    # end
    if energy_change >= 0 # Energy increased or stalled (bad step)
        # Definitely decrease the learning rate
        # state.integrator.lr = max(0.8 * state.integrator.lr, 0.00001)
        # println("Energy did not decrease, decreasing learning rate to ", state.integrator.lr)
        count[] += 1
    end
    prev_energy[] = state.energy

    # if count[]>=10
    #     state.sample_nr = min(state.sample_nr + 200,3000)
    #     # state.integrator.lr = 0.1
    #     println("Increasing sample number to ", state.sample_nr)
    #     # state.integrator.lr = state.integrator.lr *0.7

    #     # println("Setting learning rate to ", state.integrator.lr)
    #     count[] = 0
    # end

    

    if niter>20 && state.energy < best_energy[]
        best_energy[] = state.energy
        best_theta[] = deepcopy(state.θ)
        println("Saving best PEPS with energy ", best_energy[])
        count_best[] = 0
    end

    count_best[] += 1
    # if count_best[]%25==0
    #     state.θ = deepcopy(best_theta[])
    #     state.energy = best_energy[]
    #     println("Rolling back to best PEPS with energy ", best_energy[])
    # end
    # if count_best[]%50 ==0
    #     println("No improvement for 50 iterations, best PEPS with energy ", best_energy[])
    #     state.integrator.lr = state.integrator.lr*0.5
    #     println("Decreasing learning rate to ", state.integrator.lr)
    # end


    # if count_best[]==200
    #     println("No improvement for 200 iterations, stopping optimization.")
    #     return false
    # end
 
    save("save.jld2", kwargs)
end

# Timer to measure the time taken for each step and subtask
timer = TimerOutput()
# println("Starting the optimization...")
# Generate Operators for QNG
# Suppose `peps` is your PEPS
dims = (size(peps, 1), size(peps, 2))


Oks_and_Eks1 = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, ham_J1; threaded=threaded, multiproc=multiproc,shared_array=shared_array, timer) 

Oks_and_Eks2 = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, ham_J1x; threaded=threaded, multiproc=multiproc,shared_array=shared_array ,timer,peps_preconditioner = p -> basis_change!(p; gate="H"),peps_postconditioner = p -> basis_change!(p; gate="H"))

Oks_and_Eks3 = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, ham_J1y; threaded=threaded, multiproc=multiproc, shared_array=shared_array,timer,peps_preconditioner = p -> basis_change!(p; gate="ZtoY"),peps_postconditioner = p -> basis_change!(p; gate="YtoZ"))

# Setup the Integrator and Solver
integrator = QuantumNaturalGradient.Euler(lr=params[:lr], write_func=QuantumNaturalfPEPS.write!, 
    basis_func=QuantumNaturalfPEPS.basis_change!, 
    vec=QuantumNaturalfPEPS.vec,gates = ("Id","H","YtoZ")
    )
solver = QuantumNaturalGradient.EigenSolver(params[:eigencut], verbose=false)

# Initialize Parameters and Evolve
θ = QuantumNaturalGradient.Parameters(peps)

# Logger functions, will be called after each iteration and results will be saved in the misc["history"] dictionary
contract_dim(; contract_dims) = mean(contract_dims)
logger_funcs = [contract_dim]

# Evolve for a fixed (small) number of iterations as a demo
@time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve((Oks_and_Eks1,Oks_and_Eks2,Oks_and_Eks3), θ; 
        integrator, 
        verbosity=2,
        solver,
        sample_nr=params[:sample_nr],
        maxiter=params[:maxiter],
        logger_funcs,
        #misc_restart, #to restart from a previous run
        callback, 
        timer
        )
# rm("save.jld2") # remove the save file to clean up the examples directory
# stop_threads() # stop the threads, if you used async_double_layers=true
# println(trained_θ.obj)


# Z  = zeros(params[:Lx], params[:Ly]) 

# O = OpSum[]

# # for i in 1:params[:Lx], j in 1:params[:Ly]
# #     os = OpSum()
# #     os += 1.0, "Z", (i, j) 
# #     push!(O, os)
# # end

# for i in 1:params[:Lx], j in 1:params[:Ly]
#     for k in 1:params[:Lx], l in 1:params[:Ly]
#         # skip self-interaction
#         if i == k && j == l
            
#         end
#         # avoid double-counting
#         if (k < i) || (k == i && l < j)
#             os = OpSum()
#             os += 1.0, "Z", (i,j)
#             os += 1.0, "Z", (k,l)
#             pushi!(O,os)
#         end
#     end
# end

# O_loc, weights = QuantumNaturalfPEPS.get_ExpectationValue(
#     trained_θ.obj,
#     O;
#     it=1000,
#     multiproc=false
# )
# expectations = [real(sum(O_loc[:,j] .* weights) / sum(weights)) for j in 1:length(O)]

# println(expectations)




O = OpSum[]

Lx, Ly = params[:Lx], params[:Ly]

for i in 1:Lx, j in 1:Ly
    for k in i:Lx
        l_start = (k == i) ? j+1 : 1  # skip self and lower triangle
        for l in l_start:Ly
            os = OpSum()
            os .+= (1.0, "Sz", (i,j),"Sz", (k,l))
            push!(O, os)
        end
    end
end

# get expectation values
O_loc, weights = QuantumNaturalfPEPS.get_ExpectationValue(
    best_theta[].obj,
    O;
    it=10000,
    threaded=threaded,
    multiproc=multiproc
)

expectations_1d = [real(sum(O_loc[:,j] .* weights) / sum(weights)) for j in 1:length(O)]

N = Lx * Ly
corrmat = zeros(Float64, N, N)

# set diagonal = 1 (self-interaction)

function correlation_matrix_from_expectations(expectations_1d::Vector{Float64}, Lx::Int, Ly::Int)
    N = Lx * Ly
    corrmat = zeros(Float64, N, N)

    # set diagonal = 1 (self-interaction)
    for i in 1:N
        corrmat[i,i] = 1.0/4.0
    end

    count = 1
    for i in 1:Lx, j in 1:Ly
        idx1 = (i-1)*Ly + j
        for k in i:Lx
            l_start = (k == i) ? j+1 : 1
            for l in l_start:Ly
                idx2 = (k-1)*Ly + l

                val = expectations_1d[count]
                corrmat[idx1, idx2] = val
                corrmat[idx2, idx1] = val  # symmetric

                count += 1
            end
        end
    end
    return corrmat
end
corrmat = real(correlation_matrix_from_expectations(expectations_1d, Lx, Ly))
# println("Correlation matrix:")
# println(corrmat)
println("PEPS stats:")
println("Best energy from PEPS = ", best_energy[])
println("Structure factor at q = (0,0): ", structure_factor((0, 0.0), coords, corrmat, params))
println("Structure factor at q = (pi,0): ", structure_factor((pi, 0), coords, corrmat, params))
println("Structure factor at q = (0,pi): ", structure_factor((0, pi), coords, corrmat, params))
println("Structure factor at q = (pi,pi): ", structure_factor((pi, pi), coords, corrmat, params))

println("DMRG stats:")
println("Ground state energy from DMRG = $energy")
println("Structure factor at q = (0,0) from DMRG = ", structure_factor((0, 0.0), coords, corr, params))
println("Structure factor at q = (pi,0) from DMRG = ", structure_factor((pi, 0), coords, corr, params))
println("Structure factor at q = (0,pi) from DMRG =", structure_factor((0, pi), coords, corr, params))
println("Structure factor at q = (pi,pi) from DMRG = ", structure_factor((pi, pi), coords, corr, params))
# println("Structure factor at q = (0,pi/3) from DMRG = ", structure_factor((0, pi/sqrt(3)), coords, corr, params))
# println("Structure factor at q = (0,2*pi/3) from DMRG = ", structure_factor((0, 2*pi/sqrt(3)), coords, corr, params))

using Plots
plot(1:length(energy_arr), energy_arr, xlabel="Iteration", ylabel="Energy", title="Energy vs Iteration", legend=false)
savefig("energy_convergence_{lr$(params[:lr]),L$(params[:Lx]),d$(params[:bdim])}}.png")

