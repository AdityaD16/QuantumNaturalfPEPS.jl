using MPI

function generate_Oks_and_Eks_mpi(peps::AbstractPEPS,
                                                     ham_op::TensorOperatorSum;
                                                     timer=TimerOutput(),
                                                     threaded=true,
                                                     double_layer_update=update_double_layer_envs!,
                                                     reset_double_layer=true,
                                                     kwargs...)

    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    n_threads = Threads.nthreads()

    # Broadcast ham_op and Θ only once
    ham_op = MPI.bcast(ham_op, 0, comm)

    function Oks_and_Eks_(Θ::Vector{T}, sample_nr::Integer; kwargs2...) where T
        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end

        if rank == 0
            write!(peps, Θ; reset_double_layer)
            if reset_double_layer
                @timeit timer "double_layer_envs" double_layer_update(peps)
            end
        end

        # Broadcast updated peps to all ranks
        peps = MPI.bcast(peps, 0, comm)

        return @timeit timer "Oks_and_Eks" Oks_and_Eks_multiproc_sharedarrays(
            peps, ham_op, sample_nr; timer=timer,
            n_threads=n_threads, kwargs...)
    end

    # Method for Parameters wrapper
    function Oks_and_Eks_(peps_::Parameters{<:AbstractPEPS},
                          sample_nr::Integer; kwargs2...)
        peps_ = peps_.obj

        if rank == 0 && getfield(peps_, :double_layer_envs) === nothing
            @timeit timer "double_layer_envs" double_layer_update(peps_)
        end

        # Broadcast to other ranks
        peps_ = MPI.bcast(peps_, 0, comm)

        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end

        return @timeit timer "Oks_and_Eks" Oks_and_Eks_multiproc_sharedarrays(
            peps_, ham_op, sample_nr; timer=timer,
            n_threads=n_threads, kwargs...)
    end

    return Oks_and_Eks_
end



function Oks_and_Eks_mpi(peps, ham_op, sample_nr;
                                            Oks=nothing, importance_weights=true,
                                            n_threads=Threads.nthreads(),
                                            timer=TimerOutput(),
                                            kwargs...)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Split total samples across MPI ranks
    k = ceil(Int, sample_nr / nprocs)
    k_thread = ceil(Int, k / n_threads)
    k_eff = k_thread * n_threads
    sample_nr_eff = k_eff * nprocs

    nr_parameters = length(peps)
    eltype_ = eltype(peps)
    eltype_real = real(eltype_)
    seed = rand(UInt) + rank

    # Local chunk of samples for this rank
    i1 = k_eff * rank + 1
    i2 = k_eff * (rank + 1)

    # Compute locally (multithreaded)
    out_dict = Oks_and_Eks_threaded(peps, ham_op, k;
                                    importance_weights=false,
                                    seed=seed,
                                    nr_threads=n_threads,
                                    Oks=nothing,   # no SharedArray
                                    return_Oks=true, kwargs...)

    # --- Now gather results ---

    # Parameter derivatives
    Oks_local = out_dict[:Oks]      # size (k_eff, nr_params)
    Oks_all = MPI.gather(Oks_local, 0, comm)

    # Energies etc.
    Eks_all        = MPI.gather(out_dict[:Eks], 0, comm)
    logψs_all      = MPI.gather(out_dict[:logψs], 0, comm)
    samples_all    = MPI.gather(out_dict[:samples], 0, comm)
    weights_all    = MPI.gather(out_dict[:weights], 0, comm)
    contract_dims_all = MPI.gather(out_dict[:contract_dims], 0, comm)

    if rank != 0
        return nothing
    end

    # Concatenate on rank 0
    Oks = reduce(vcat, Oks_all)
    Eks = reduce(vcat, Eks_all)
    logψs = reduce(vcat, logψs_all)
    samples = reduce(vcat, samples_all)
    logpcs = reduce(vcat, weights_all)
    contract_dims = reduce(vcat, contract_dims_all)

    if importance_weights
        weights = compute_importance_weights(logψs, logpcs)
    else
        weights = logpcs
    end

    return Dict(:Oks => Oks,
                :Eks => Eks,
                :logψs => logψs,
                :samples => samples,
                :weights => weights,
                :contract_dims => contract_dims)
end
