using Random, LinearAlgebra, JLD2, TimerOutputs, ITensors, Distributed
JLD2.rconvert(T::Type, x::Tuple{Symbol}) = (x[1], x[1])
using QuantumNaturalGradient
using QuantumNaturalfPEPS
const QNG = QuantumNaturalGradient

# Setup: Processes, Threads, and BLAS Settings
const NR_PROCS   = 9
const NR_THREADS = 8
addprocs(NR_PROCS, exeflags="-t $NR_THREADS")

@everywhere begin
    using QuantumNaturalfPEPS
    using LinearAlgebra
    BLAS.set_num_threads(1)  # Prevent nested BLAS threading in each process
end

# Main process BLAS setting (e.g., for double layers)
BLAS.set_num_threads(16)

# Simulation Parameters
L    = 4
J1   = 2 * cos(0.06*pi) * cos(0.14*pi)
J2   = 2 * cos(0.06*pi) * sin(0.14*pi)
λ    = 2 * sin(0.06*pi)

# Define a parameter dictionary for easy adjustments
params = Dict{Symbol, Any}(
    :T              => ComplexF64,
    :seed           => 1,
    :Lx             => L,
    :Ly             => L,
    :bond_dim       => 2,
    :sample_nr      => 1000,
    :lr             => 0.05,
    :J1             => J1,
    :J2             => J2,
    :lambda         => λ,
    :eigencut       => 1e-4,
    :contract_cutoff=> 1e-4,
    :sample_cutoff  => 1e-3,
    :contract_dim   => 100,
    :maxiter        => 4000,
    :α_init         => 3.0
)
println("Simulation parameters: ", params)

# Set random seed for reproducibility
Random.seed!(params[:seed])

save_file = "CSL.jld2"

# ---------------------------------------------------------------------------
# Hilbert Space and PEPS Initialization
# ---------------------------------------------------------------------------
hilbert = siteinds("S=1/2", params[:Lx], params[:Ly])

# For clarity, use explicit names for dimension parameters
bond_dim            = params[:bond_dim]
double_contract_dim = bond_dim      # alias for double-layer contraction

peps = PEPS(params[:T], hilbert;
    bond_dim           = bond_dim,
    sample_dim         = params[:contract_dim],
    sample_cutoff      = params[:sample_cutoff],
    double_contract_dim= double_contract_dim,
    contract_dim       = params[:contract_dim],
    contract_cutoff    = params[:contract_cutoff],
    show_warning       = true
)

# Multiply the spectrum of the PEPS by a power-law factor to make it contractible
QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps, params[:α_init])

# ---------------------------------------------------------------------------
# Hamiltonian:  J1 S.S  +  J2 S.S  +  i*lambda*(P - P^dagger)   [PRB 111, 195120 Eq. 26]
# ---------------------------------------------------------------------------
# Built by the package rather than by local copies of the helpers: the plaquette
# operator is easy to get wrong (index ordering, the inverse cycle, and keeping
# TensorOperatorSum's precomputed `terms` in sync), and every such mistake makes
# the chiral term silently evaluate to zero, i.e. a plain J1-J2 run.
ham_op = QuantumNaturalfPEPS.hamiltonain_CSL(hilbert, params[:J1], params[:J2], params[:lambda])

n_chiral = 2 * (params[:Lx] - 1) * (params[:Ly] - 1)
@assert length(ham_op.terms) == length(ham_op.tensors)   # estimator reads `terms`
println("Hamiltonian: $(length(ham_op.terms)) terms ($n_chiral chiral), eltype $(eltype(ham_op))")

# Setup for Evolution: Timer, Operators, Integrator, Solver, Logger, etc.
timer = TimerOutput()
Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, ham_op; timer, threaded=true, multiproc=true)

integrator = QNG.Euler(lr=params[:lr],
                       use_clipping=false,
                       clip_norm=0.03 * params[:Lx] * params[:Ly] * params[:bond_dim]^4)
solver = QNG.EigenSolver(params[:eigencut], verbose=true)

# Convert the PEPS to a Parameters object
θ = QuantumNaturalGradient.Parameters(peps)

# Define a logger function to record contract dimensions (or any custom data)
history_contract_dims(; contract_dims) = contract_dims
logger_funcs = [history_contract_dims]

# Callback function to save data at each iteration
function callback(; niter=1, kwargs...)
    data = Dict{String, Any}(string(key) => v for (key, v) in kwargs)
    data["params"]     = params
    data["solver"]     = solver
    data["integrator"] = integrator
    data["peps"]       = peps
    data["timer"]      = timer
    save(save_file, data)
end

# Optionally restart from a saved state if it exists
misc_restart = nothing
if isfile(save_file)
    println("Restarting from saved file: $save_file")
    d = load(save_file)
    θ          = d["model"]
    solver     = d["solver"]
    integrator = d["integrator"]
    misc_restart = d["misc"]
    @show misc_restart.niter[end]
end

# Run the Evolution Process
@time loss_value, trained_θ, misc = QNG.evolve(Oks_and_Eks, θ;
    integrator,
    verbosity=2,
    solver,
    sample_nr = params[:sample_nr],
    maxiter   = params[:maxiter],
    callback,
    timer,
    misc_restart,
    logger_funcs
)
rm(save_file)  # Clean up example folder by removing the save file