# Heisenberg model with particle-number (total Sz) conserving sampling.

using LinearAlgebra
using Statistics
using Random
using ITensors, ITensorMPS
using QuantumNaturalGradient
using QuantumNaturalfPEPS
using Distributed
const QNP = QuantumNaturalfPEPS

# Initialize 2 processes with 8 threads each, assuming you have a 16 core cpu
nr_procs, nr_threads = 2, 8
addprocs(nr_procs, exeflags="-t $nr_threads")

@everywhere begin
    using QuantumNaturalfPEPS
    using LinearAlgebra
    BLAS.set_num_threads(1) # Each thread should only use one Blas thread
end
BLAS.set_num_threads(16) # For computing the double layers

Random.seed!(1)

# Simulation Parameters
params = Dict(
    :T              => Float64,
    :Lx             => 4,
    :Ly             => 4,
    :bdim           => 2,
    :lr             => 0.05,
    :J              => 1.0,
    :Sz             => 0,      # target sector: total Sz
    :eigencut       => 1e-4,
    :contract_cutoff=> 1e-4,
    :sample_cutoff  => 1e-3,
    :contract_dim   => 100,
    :maxiter        => 50,
    :α_init         => 2.0,
    :sample_nr      => 500,
)
N = params[:Lx] * params[:Ly]

# Define the Lattice and PEPS
hilbert = siteinds("S=1/2", params[:Lx], params[:Ly])
peps = PEPS(params[:T], hilbert; bond_dim=params[:bdim], show_warning=true,
    contract_cutoff=params[:contract_cutoff],
    contract_dim=params[:contract_dim],
    sample_cutoff=params[:sample_cutoff],
)

# Multiply the spectrum of the PEPS by a power-law factor to make it contractible
QNP.multiply_algebraic_spectrum!(peps, params[:α_init])

# Construct the Hamiltonian  J Σ (Sz Sz + (S+ S- + S- S+)/2)
ham = OpSum()
function add_bond!(ham, J, a, b)
    ham .+= (J, "Sz", a, "Sz", b)
    ham .+= (J / 2, "S+", a, "S-", b)
    ham .+= (J / 2, "S-", a, "S+", b)
end
for i in 1:params[:Lx], j in 1:params[:Ly]
    j < params[:Ly] && add_bond!(ham, params[:J], (i, j), (i, j+1))
    i < params[:Lx] && add_bond!(ham, params[:J], (i, j), (i+1, j))
end

# The sector constraint: caps on the number of ↑ (value 0) and ↓ (value 1) spins.
# For Sz = 0 on 16 sites this is [8, 8], i.e. exactly 8 up and 8 down spins.
max_counts = QNP.sz_sector_counts(peps, params[:Sz])
println("Sampling in the total Sz = $(params[:Sz]) sector, max_counts = $max_counts")

# Generate Operators for QNG; max_counts is forwarded to every sample drawn
Oks_and_Eks = QNP.generate_Oks_and_Eks(peps, ham; threaded=true, multiproc=true, max_counts)

# Setup the Integrator and Solver
integrator = QuantumNaturalGradient.Euler(lr=params[:lr])
solver = QuantumNaturalGradient.EigenSolver(params[:eigencut])

# Initialize Parameters (they share their tensors with `peps`)
θ = QuantumNaturalGradient.Parameters(peps)

# Print the energy per site after each iteration
function callback(; state, niter=1, kwargs...)
    E = real(state.energy)
    σ2 = state.history.var_energy[end]
    println("iter $niter: E = $(round(E; digits=5)),  E/site = $(round(E / N; digits=5)),  Var(E) = $(round(σ2; sigdigits=3))")
    flush(stdout)
end

@time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve(Oks_and_Eks, θ;
        integrator,
        verbosity=1,
        solver,
        sample_nr=params[:sample_nr],
        maxiter=params[:maxiter],
        callback,
        )

# Check: constrained samples all carry total Sz = 0, while plain samples of the
# same state spread over other sectors (the samples the constraint saves).
# Note: a state optimized with the constraint should also be EVALUATED with it,
# since nothing in the optimization penalizes its weight outside the sector.
QNP.update_double_layer_envs!(peps)
total_Sz(S) = sum(v -> v == 0 ? 0.5 : -0.5, S)
nr_check = 100
Sz_constrained = [total_Sz(QNP.get_sample(peps; max_counts)[1]) for _ in 1:nr_check]
Sz_plain       = [total_Sz(QNP.get_sample(peps)[1]) for _ in 1:nr_check]
println("Constrained samples in the Sz = $(params[:Sz]) sector: $(count(==(params[:Sz]), Sz_constrained)) / $nr_check")
println("Plain samples in the Sz = $(params[:Sz]) sector:       $(count(==(params[:Sz]), Sz_plain)) / $nr_check")
println("Final energy per site: $(real(loss_value) / N)")
