function hamiltonain_J1J2(J1, J2, Lx, Ly; operators=["X", "Y", "Z"])
    ham_J1J2 = OpSum()
    for i in 1:Lx, j in 1:Ly, t in operators
        # J1
        if j != Ly
            ham_J1J2 += (J1, t, (i,j), t, (i,j+1))
        end
        if i != Lx
            ham_J1J2 += (J1, t, (i,j), t, (i+1,j))
        end

        # J2
        if i != Lx && j != Ly
            ham_J1J2 += (J2, t, (i,j), t, (i+1,j+1))
            ham_J1J2 += (J2, t, (i+1,j), t, (i,j+1))
        end
    end
    return ham_J1J2
end


# Define Helper Functions for P Operators
# The ITensor built below carries its indices in the order
# (s1', s2', s3', s4', s1, s2, s3, s4) -- all bras first, then all kets -- so the
# bra slots must hold the cyclically shifted ket labels.  `inverse=true` gives
# P^dagger, the opposite cycle.
function P_matrix(factor; inverse=false)
    # Create an 8-index tensor and fill specific indices with the factor.
    P = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2, 2)
    for i1 in 1:2, i2 in 1:2, i3 in 1:2, i4 in 1:2
        if inverse
            P[i2, i3, i4, i1,  i1, i2, i3, i4] = factor  # |s1s2s3s4> -> |s2s3s4s1>
        else
            P[i4, i1, i2, i3,  i1, i2, i3, i4] = factor  # |s1s2s3s4> -> |s4s1s2s3>
        end
    end
    return P
end

function P_operator(hilbert, spins; P=nothing, factor=1, inverse=false)
    # Generate the ITensor operator for a given set of spins.
    if P === nothing
        P = P_matrix(factor; inverse)
    end
    inds = [hilbert[s]' for s in spins]  # use prime on physical indices
    append!(inds, [hilbert[s] for s in spins])
    return ITensor(P, inds)
end

function add_P_operators!(ham_op, hilbert, Lx, Ly, factor; inverse=false)
    sites = Int[]
    for i in 1:(Lx-1)
        for j in 1:(Ly-1)
            # Define the sites of the plaquette
            push!(sites, i   + (j-1)*Lx)
            push!(sites, i+1 + (j-1)*Lx)
            push!(sites, i+1 + (j)*Lx)
            push!(sites, i   + (j)*Lx)
            # Create and add the operator
            P_op = P_operator(hilbert, sites; factor=factor, inverse=inverse)
            push!(ham_op.tensors, P_op)
            push!(ham_op.sites, copy(sites))
            # TensorOperatorSum evaluates Eloc from its PRECOMPUTED `terms` list,
            # not from `tensors`.  Appending to tensors/sites alone leaves the new
            # operator invisible to the estimator (it silently contributes zero).
            push!(ham_op.terms, QuantumNaturalGradient._precompute_term(
                      eltype(ham_op), P_op, copy(sites), ham_op.hilbert[:]))
            empty!(sites)
        end
    end
end

function hamiltonain_CSL(hilbert, J1, J2, lambda; kwargs...)
    ham_J1J2 = hamiltonain_J1J2(J1/4, J2/4, size(hilbert)...; kwargs...)

    # Create the tensor operator for the Hamiltonian
    tn_sum = QuantumNaturalGradient.TensorOperatorSum(ham_J1J2, hilbert)
    @assert eltype(tn_sum) <: Complex "CSL needs a complex TensorOperatorSum; got $(eltype(tn_sum))"

    # Chiral term  i*lambda*(P - P^dagger).  The second call must build the INVERSE
    # cycle, otherwise both contributions are the same operator and cancel exactly.
    add_P_operators!(tn_sum, hilbert, size(hilbert)...,  im * lambda)
    add_P_operators!(tn_sum, hilbert, size(hilbert)..., -im * lambda; inverse=true)
    return tn_sum
end