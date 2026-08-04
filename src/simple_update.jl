function construct_evolution_gates(ham_op,dt; mode="imag")
    gate_op = deepcopy(ham_op)
    Nx = size(gate_op)[1]
    Ny = size(gate_op)[2]
    coord_to_2d(i, Nx) = begin
        y = div(i-1, Nx) + 1     # row indexs
        x = mod(i-1, Nx) + 1     # column index
        return x, y
    end
    if (mode == "real")
        dt = im * dt
    end
    for n in 1:(length(gate_op.tensors))
        if length(gate_op.sites[n])==2
            x1,y1 = coord_to_2d(gate_op.sites[n][1], Nx)
            x2,y2 = coord_to_2d(gate_op.sites[n][2], Nx)
            if abs(x1 - x2) == 1 && abs(y1 - y2) == 1
                # diagonal terms are applied twice (once via each third site)
                gate_op.tensors[n] = exp(-(dt/2) * gate_op.tensors[n])
            else
                gate_op.tensors[n] = exp(-dt * gate_op.tensors[n])
            end
        else
            gate_op.tensors[n] = exp(-dt * gate_op.tensors[n])
        end
     
    end
    return gate_op
end

@inline function find_bond_tensor(peps, site1, site2, Sx, Sy)
    x1,y1 = site1
    x2,y2 = site2

    l = commonind(peps[x1,y1], peps[x2,y2])
    @assert l != nothing "No common index between sites ($x1,$y1) and ($x2,$y2)!"

    if x1==x2
        S = itensor(diagm((Sy[x1,min(y1,y2),:])) , l, l')
    else
        S = itensor(diagm((Sx[min(x1,x2),y1,:])) , l, l')
    end
    return S
end

@inline function update_bond_tensor!(site1,site2,Sx,Sy,new_S)
    x1,y1 = site1
    x2,y2 = site2

    if x1==x2
        Sy[x1,min(y1,y2),:] = diag(new_S)
    else
        Sx[min(x1,x2),y1,:] = diag(new_S)
    end
    return Sx, Sy
end

function collect_bond_matrices(peps,site,Sx,Sy;exception = (-1,-1))
    x,y = site
    bonds = []
    S_vals = []
    Sinv_vals = []
    if x > 1 && exception != (x-1,y)
        push!(bonds, commonind(peps[x-1,y], peps[x,y]))
        s,s_inv = QuantumNaturalfPEPS.Smatrix(Sx[x-1,y,:], bonds[end],cutoff=1e-12)
        push!(S_vals, s)
        push!(Sinv_vals, s_inv)
    end
    if x < size(peps,1) && exception != (x+1,y)
        push!(bonds, commonind(peps[x,y], peps[x+1,y]))
        s,s_inv = QuantumNaturalfPEPS.Smatrix(Sx[x,y,:], bonds[end],cutoff=1e-12)
        push!(S_vals, s)
        push!(Sinv_vals, s_inv)
    end
    if y > 1 && exception != (x,y-1)
        push!(bonds, commonind(peps[x,y-1], peps[x,y]))
        s,s_inv = QuantumNaturalfPEPS.Smatrix(Sy[x,y-1,:], bonds[end],cutoff=1e-12)
        push!(S_vals, s)
        push!(Sinv_vals, s_inv)
    end
    if y < size(peps,2) && exception != (x,y+1)
        push!(bonds, commonind(peps[x,y], peps[x,y+1]))
        s,s_inv = QuantumNaturalfPEPS.Smatrix(Sy[x,y,:], bonds[end],cutoff=1e-12)
        push!(S_vals, s)
        push!(Sinv_vals, s_inv)
    end
    return bonds, S_vals, Sinv_vals
end


@inline function two_qubit_gate_application!(peps, site1, site2, gate, Sx, Sy, su_bdim;reduced=true,reversed=false)

    x1,y1 = site1
    x2,y2 = site2
    dx, dy = abs(x1 - x2), abs(y1 - y2)
    # Only nearest neighbors (dx+dy==1) and the diagonal next-nearest neighbors
    # (dx==dy==1) are supported. Note that dx+dy==2 alone is not enough: it also
    # matches the straight distance-2 pairs (i,j)-(i+2,j), which have no shared
    # bond and no common third site.
    (dx + dy == 1 || (dx == 1 && dy == 1)) || error("Only nearest-neighbor and diagonal next-nearest-neighbor terms are supported on the square lattice; got sites ($x1,$y1) and ($x2,$y2).")
    if dx == 1 && dy == 1
         diagonal_gate_application!(peps, site1, site2, gate, Sx, Sy, su_bdim; reversed=reversed)

    elseif reduced
         reduced_nn_gate_application!(peps, site1, site2, gate, Sx, Sy, su_bdim)
    else
        full_nn_gate_application!(peps, site1, site2, gate, Sx, Sy, su_bdim)
    end
    return peps, Sx, Sy
end
two_qubit_gate_application(peps, site1, site2, gate, Sx, Sy, su_bdim;kwargs...) = two_qubit_gate_application!(deepcopy(peps), site1, site2, gate, deepcopy(Sx), deepcopy(Sy), su_bdim; kwargs...)


function full_nn_gate_application!(peps, site1, site2, gate, Sx, Sy, su_bdim)

    x1,y1 = site1
    x2,y2 = site2

    bonds1, S_vals1, Sinv_vals1 = collect_bond_matrices(peps, (x1,y1), Sx, Sy, exception=(x2,y2))
    bonds2, S_vals2, Sinv_vals2 = collect_bond_matrices(peps, (x2,y2), Sx, Sy, exception=(x1,y1))
    l = commonind(peps[x1,y1], peps[x2,y2])
    @assert l != nothing "No common index between sites ($x1,$y1) and ($x2,$y2)!"

    T1 = peps[x1,y1]
    T2 = peps[x2,y2]

    # Absorb S matrices into tensors
    T1 = apply(T1, S_vals1)
    T2 = apply(T2, S_vals2)

    # Find bond tensor
    S = find_bond_tensor(peps, (x1,y1), (x2,y2), Sx, Sy)

    # Apply gate
    A = apply(T1, S)
    A = apply(A, T2)
    @assert length(commoninds(A,gate)) == 2 "Error: common indices between A and gate tensor is not 2!"
    A = apply(gate, A)  # gate acts on A: apply(A, gate) would contract the transpose

    left_indices = inds(peps[x1,y1])
    left_indices = setdiff(left_indices, (l,))
    u, s, v = svd(A,left_indices;maxdim=su_bdim)
    s = s / norm(diag(s))
    bL = inds(s)[1]
    bR = inds(s)[2]
    u = replaceind(u, bL,l)
    v = replaceind(v, bR,l)

    # Update S matrices
    update_bond_tensor!((x1,y1),(x2,y2),Sx,Sy,s)

    # Absorb inverse S matrices back into tensors
    u = apply(u, Sinv_vals1)
    v = apply(v, Sinv_vals2)

    # Update PEPS tensors
    peps[x1,y1] = u
    peps[x2,y2] = v

    return peps, Sx, Sy

end

function reduced_nn_gate_application!(peps,site1, site2, gate, Sx, Sy, su_bdim)

    x1,y1 = site1
    x2,y2 = site2

    bonds1, S_vals1, Sinv_vals1 = collect_bond_matrices(peps, (x1,y1), Sx, Sy, exception=(x2,y2))
    bonds2, S_vals2, Sinv_vals2 = collect_bond_matrices(peps, (x2,y2), Sx, Sy, exception=(x1,y1))
    l = commonind(peps[x1,y1], peps[x2,y2])
    @assert l != nothing "No common index between sites ($x1,$y1) and ($x2,$y2)!"

    T1 = peps[x1,y1]
    T2 = peps[x2,y2]
    
    # Absorb S matrices into tensors
    T1 = apply(T1,S_vals1)
    T2 = apply(T2, S_vals2)

    # QR decomposition 
    left_Q_indices = setdiff(inds(peps[x1,y1]), (l,siteind(peps,x1,y1)))
    right_Q_indices = setdiff(inds(peps[x2,y2]), (l,siteind(peps,x2,y2)))
    Q1, R1 = qr(T1, left_Q_indices)
    Q2, R2 = qr(T2, right_Q_indices)

    # Find bond tensor
    S = find_bond_tensor(peps, (x1,y1), (x2,y2), Sx, Sy)

    # Apply gate
    A = apply(R1, S)
    A = apply(A, R2)

    @assert length(commoninds(A,gate)) == 2 "Error: common indices between A and gate tensor is not 2!"
    A = apply(gate, A)  # gate acts on A: apply(A, gate) would contract the transpose

    indices_U = (commoninds(Q1,R1),siteind(peps,x1,y1))
    u,s,v = svd(A,indices_U;maxdim=su_bdim)
    s = s / norm(diag(s))
    bL = inds(s)[1]
    bR = inds(s)[2]
    u = replaceind(u, bL,l)
    v = replaceind(v, bR,l)

    # Update S matrices
    update_bond_tensor!((x1,y1),(x2,y2),Sx,Sy,s)

    #Absorb Q matrices back into tensors
    u = apply(Q1, u)
    v = apply(Q2, v)

    # Absorb inverse S matrices back into tensors
    u = apply(u, Sinv_vals1)
    v = apply(v, Sinv_vals2)

    # Update PEPS tensors
    peps[x1,y1] = u
    peps[x2,y2] = v

    return peps, Sx, Sy
end

function diagonal_gate_application!(peps,site1,site2,site3,gate,Sx,Sy,su_bdim)
    
    x1,y1 = site1
    x2,y2 = site2
    x3,y3 = site3

    bonds1, S_vals1, Sinv_vals1 = collect_bond_matrices(peps, (x1,y1), Sx, Sy, exception=(x3,y3))
    bonds2, S_vals2, Sinv_vals2 = collect_bond_matrices(peps, (x2,y2), Sx, Sy, exception=(x3,y3))
    bonds3, S_vals3, Sinv_vals3 = collect_bond_matrices(peps, (x3,y3), Sx, Sy)
    l1 = commonind(peps[x1,y1], peps[x3,y3])
    l2 = commonind(peps[x2,y2], peps[x3,y3])
    @assert l1 != nothing "No common index between sites ($x1,$y1) and ($x3,$y3)!"
    @assert l2 != nothing "No common index between sites ($x2,$y2) and ($x3,$y3)!"

    S_vals3 = [S_vals3[i] for i in 1:length(S_vals3) if bonds3[i] != l1 && bonds3[i] != l2]
    Sinv_vals3 = [Sinv_vals3[i] for i in 1:length(Sinv_vals3) if bonds3[i] != l1 && bonds3[i] != l2]

    T1 = peps[x1,y1]
    T2 = peps[x2,y2]
    T3 = peps[x3,y3]

    # Absorb S matrices into tensors
    T1 = apply(T1, S_vals1)
    T2 = apply(T2, S_vals2)
    T3 = apply(T3, S_vals3)

    # QR decomposition
    indices_Q1 = setdiff(inds(peps[x1,y1]), (l1,siteind(peps,x1,y1)))
    indices_Q2 = setdiff(inds(peps[x2,y2]), (l2,siteind(peps,x2,y2)))
    indices_Q3 = setdiff(inds(peps[x3,y3]), (l1,l2,siteind(peps,x3,y3)))
    Q1, R1 = qr(T1, indices_Q1)
    Q2, R2 = qr(T2, indices_Q2)
    Q3, R3 = qr(T3, indices_Q3)

    # Find bond tensor
    S1 = find_bond_tensor(peps, (x1,y1), (x3,y3), Sx, Sy)
    S2 = find_bond_tensor(peps, (x2,y2), (x3,y3), Sx, Sy)

    # Apply gate
    A = apply(R1, S1)
    A = apply(A, R3)
    A = apply(A, S2)
    A = apply(A, R2)
    @assert length(commoninds(A,gate)) == 2 "Error: common indices between A and gate tensor is not 2!"
    A = apply(gate, A)  # gate acts on A: apply(A, gate) would contract the transpose

    # Apply vertical and horizontal SVDs
    indices_U1 = (commoninds(Q1,R1),siteind(peps,x1,y1))
    u1,s1,A = svd(A,indices_U1;maxdim=su_bdim)
    s1 = s1 / norm(diag(s1))
    bL1 = inds(s1)[1]
    bR1 = inds(s1)[2]
    u1 = replaceind(u1, bL1,l1)
    A = replaceind(A, bR1,l1)
    indices_U2 = (commoninds(Q2,R2),siteind(peps,x2,y2))
    u2,s2,v2 = svd(A,indices_U2;maxdim=su_bdim)
    s2 = s2 / norm(diag(s2))
    bL2 = inds(s2)[1]
    bR2 = inds(s2)[2]
    u2 = replaceind(u2, bL2,l2)
    v2 = replaceind(v2, bR2,l2)

    # Update S matrices
    update_bond_tensor!((x1,y1),(x3,y3),Sx,Sy,s1)
    update_bond_tensor!((x2,y2),(x3,y3),Sx,Sy,s2)

    # Absorb Q matrices back into tensors
    u1 = apply(Q1, u1)
    u2 = apply(Q2, u2)
    v2 = apply(Q3, v2)

    # Absorb inverse S matrices back into tensors
    u1 = apply(u1, Sinv_vals1)
    u2 = apply(u2, Sinv_vals2)
    v2 = apply(v2, Sinv_vals3)

    # Update PEPS tensors
    peps[x1,y1] = u1
    peps[x2,y2] = u2
    peps[x3,y3] = v2 

    return peps, Sx, Sy
end


@inline function diagonal_gate_application!(peps,site1,site2,gate,Sx,Sy,su_bdim;reversed=false)

    x1,y1 = site1
    x2,y2 = site2

    @assert abs(x1 - x2) == 1 && abs(y1 - y2) == 1 "Supports only diagonal interaction on square lattice!"
    # Find the third site involved in the diagonal interaction. The half gate is
    # routed once through each of the two corners; `reversed` swaps that order so
    # that a reversed sweep is the exact mirror image of a forward one.
    corners = ((x1,y2), (x2,y1))
    reversed && (corners = reverse(corners))
    for corner in corners
        peps, Sx, Sy = diagonal_gate_application!(peps, (x1,y1),(x2,y2),corner, gate, Sx, Sy, su_bdim)
    end
    return peps, Sx, Sy

end





function gate_application!(peps,gate_op,Sx,Sy,su_bdim;reversed=false,kwargs...)
    Nx = size(gate_op)[1]
    Ny = size(gate_op)[2]

    coord_to_2d(i, Nx) = begin
        y = div(i-1, Nx) + 1     # row index
        x = mod(i-1, Nx) + 1     # column index
        return x, y
    end

    gate_order = reversed ? reverse(eachindex(gate_op.tensors)) : eachindex(gate_op.tensors)
    for j in gate_order
        # Single-site gate
        if length((gate_op.sites[j])) ==1
            z = gate_op.sites[j][1]
            x, y = coord_to_2d(z, Nx)
            peps[x,y] = apply(gate_op.tensors[j], peps[x,y])
            continue
        end
        # Two-site gate
        z1, z2  = gate_op.sites[j]
        x1, y1 = coord_to_2d(z1, Nx)
        x2, y2 = coord_to_2d(z2, Nx)
        # @assert abs(x1-x2) + abs(y1-y2) == 1 "Supports only nearest-neighbor interaction on square lattice!"
        two_qubit_gate_application!(peps, (x1,y1), (x2,y2), gate_op.tensors[j], Sx, Sy, su_bdim; reversed=reversed, kwargs...)
    end

    return peps, Sx, Sy
end
gate_application(peps,gate_op,Sx,Sy,su_bdim;kwargs...) = gate_application!(deepcopy(peps),gate_op,deepcopy(Sx),deepcopy(Sy),su_bdim;kwargs...)

        
@inline function absorb_S_matrices_into_peps!(peps, Sx, Sy;inverse=false)
    Nx = size(peps,1)
    Ny = size(peps,2)
    for x in 1:Nx, y in 1:Ny
        S_vals, S_invs = QuantumNaturalfPEPS.get_peps_sqrtS(peps, Sx, Sy, x, y)
        if inverse
            peps[x,y] = apply(peps[x,y],S_invs)
        else
            peps[x,y] = apply(peps[x,y],S_vals)
        end
    end
    return peps
end
absorb_S_matrices_into_peps(peps,Sx,Sy) = absorb_S_matrices_into_peps!(deepcopy(peps),Sx,Sy)

@inline function truncate_bond!(peps, site1, site2, S, new_dim)
    x1,y1 = site1
    x2,y2 = site2
    l = commonind(peps[x1,y1], peps[x2,y2])
    @assert l !== nothing "No common index between sites ($x1,$y1) and ($x2,$y2)!"
    dim(l) <= new_dim && return 0.0

    # In the super-orthogonal gauge the bond basis is the (approximate) Schmidt basis
    # and S is sorted in descending order, so keeping the first new_dim states is just
    # a slice of both adjacent tensors.
    l_new = Index(new_dim; tags=tags(l))
    P = ITensor(eltype(peps), l, l_new)
    for i in 1:new_dim
        P[l => i, l_new => i] = one(eltype(peps))
    end
    peps[x1,y1] = peps[x1,y1] * P
    peps[x2,y2] = peps[x2,y2] * P

    return sum(abs2, @view S[new_dim+1:end]) / sum(abs2, S)
end

"""
    truncate_bond_dim!(peps, new_bond_dim; nr_steps=1, reset_dims=true, k=100, error=1e-6, verbose=true)

Truncate `peps` from its bond dimension D down to `new_bond_dim` < D.

The PEPS is brought into the super-orthogonal (Vidal) gauge, in which the singular
values of a bond approximate its Schmidt spectrum; the smallest ones are then dropped
by slicing the bond index of the two adjacent tensors. On a lattice with loops the
super-orthogonal gauge is only the belief-propagation fixed point, so this is the same
quasi-optimal, local approximation that the simple update itself makes -- not a
globally optimal truncation. `nr_steps > 1` lowers the bond dimension gradually and
re-gauges in between, which is more accurate for large reductions.

The contraction and sampling dimensions of `peps` were sized for the old bond
dimension. With `reset_dims=true` (the default) they are reset to the constructor
defaults for `new_bond_dim` (`sample_dim = double_contract_dim = new_bond_dim`,
`contract_dim = 3*new_bond_dim`) and the change is reported with `@info`; pass
`reset_dims=false` to keep hand-tuned values.

Returns `(peps, discarded_weight)` where `discarded_weight` is the largest
`sum_{i>new_bond_dim} λ_i^2` encountered on any bond.
"""
function truncate_bond_dim!(peps, new_bond_dim; nr_steps=1, reset_dims=true, k=100, error=1e-6, verbose=true)
    D = peps.bond_dim
    hasqns(peps[1,1]) && throw(ArgumentError("truncate_bond_dim! is not implemented for QN-symmetric PEPS"))
    new_bond_dim >= 1 || throw(ArgumentError("new_bond_dim ($new_bond_dim) must be at least 1"))
    new_bond_dim <= D || throw(ArgumentError("new_bond_dim ($new_bond_dim) must not exceed the current bond dimension ($D)"))
    new_bond_dim == D && return peps, 0.0

    schedule = unique(round.(Int, range(D, new_bond_dim; length=nr_steps+1)))[2:end]
    max_discarded = 0.0
    for d in schedule
        Sx, Sy, _, _, _ = QuantumNaturalfPEPS.super_orthonormalization!(peps; k=k, error=error)
        discarded = 0.0
        for x in 1:size(peps,1), y in 1:size(peps,2)
            if x < size(peps,1)
                discarded = max(discarded, truncate_bond!(peps, (x,y), (x+1,y), Sx[x,y,:], d))
            end
            if y < size(peps,2)
                discarded = max(discarded, truncate_bond!(peps, (x,y), (x,y+1), Sy[x,y,:], d))
            end
        end
        peps.bond_dim = d
        max_discarded = max(max_discarded, discarded)
        verbose && println("Truncated to bond dimension $d, largest discarded weight: $discarded")
    end
    # Re-gauge at the new bond dimension and drop the now stale environments.
    QuantumNaturalfPEPS.super_orthonormalization!(peps; k=k, error=error)
    peps.double_layer_envs = nothing

    if reset_dims
        old_dims = (peps.sample_dim, peps.contract_dim, peps.double_contract_dim)
        peps.sample_dim = new_bond_dim
        peps.contract_dim = 3*new_bond_dim
        peps.double_contract_dim = new_bond_dim
        @info "truncate_bond_dim!: bond_dim $D -> $new_bond_dim; reset (sample_dim, contract_dim, double_contract_dim) $old_dims -> $((peps.sample_dim, peps.contract_dim, peps.double_contract_dim)) to the constructor defaults. Pass reset_dims=false to keep the previous values."
    end

    return peps, max_discarded
end
truncate_bond_dim(peps, new_bond_dim; kwargs...) = truncate_bond_dim!(deepcopy(peps), new_bond_dim; kwargs...)

@inline function relative_difference( Sx_,Sy_,Sx,Sy)
    # Compared per bond: Sx/Sy are (Nx-1, Ny, D) / (Nx, Ny-1, D) arrays, so the
    # spectrum of one bond is a slice along the last dimension. Comparing single
    # singular values instead would be dominated by the smallest Schmidt values
    # and would divide by zero on empty ones.
    relative_diff = 0.0
    nr_bonds = 0
    for (S_, S) in ((Sx_, Sx), (Sy_, Sy))
        for i in axes(S_, 1), j in axes(S_, 2)
            s1 = @view S_[i, j, :]
            s2 = @view S[i, j, :]
            n1 = norm(s1)
            n1 == 0 && continue
            relative_diff += norm(s2 .- s1) / n1
            nr_bonds += 1
        end
    end

    return nr_bonds == 0 ? 0.0 : relative_diff / nr_bonds
end


@inline function check_su_bdim(peps, su_bdim)
    # Sx/Sy are allocated with a fixed last dimension peps.bond_dim, and every bond
    # index of the PEPS has that dimension, so a truncation to a different su_bdim
    # cannot be written back into either.
    su_bdim == peps.bond_dim || throw(ArgumentError("su_bdim ($su_bdim) must equal peps.bond_dim ($(peps.bond_dim)); grow or shrink the PEPS bond dimension instead."))
    return nothing
end

"""
    simple_update!(peps, ham; iter, dt, su_bdim, convergence_cutoff, order, verbose, kwargs...)

Imaginary-time simple update of `peps` with the Hamiltonian `ham`, in steps of `dt`.

`order` selects the Trotter decomposition of one step. `order=1` sweeps the gate list
once with the full `dt`, giving an O(dt) error per unit imaginary time. `order=2` (the
default) uses the symmetric decomposition: the gates are built with `dt/2` and the list
is swept forward and then backward, which costs two sweeps per step but reduces the
error to O(dt^2) and in practice allows a ~10x larger `dt`.
"""
function simple_update!(peps,ham;iter=100,dt=0.01,su_bdim=nothing,convergence_cutoff = 1e-11,order=2,verbose=true,kwargs...) #Todo: add variable su_bdim
    su_bdim === nothing && (su_bdim = peps.bond_dim)
    check_su_bdim(peps, su_bdim)
    order in (1, 2) || throw(ArgumentError("order must be 1 (first-order Trotter) or 2 (symmetric Trotter), got $order"))
    ham_op = QuantumNaturalGradient.TensorOperatorSum(ham, siteinds(peps))
    # For the symmetric decomposition each half sweep carries dt/2, so that one
    # iteration still advances the imaginary time by dt.
    gate_op = construct_evolution_gates((ham_op), order == 2 ? dt/2 : dt)
    Sx,Sy,_,_,_ = QuantumNaturalfPEPS.super_orthonormalization!(peps; k=100, error=1e-6)
    for i in 1:iter
        absorb_S_matrices_into_peps!(peps, Sx, Sy; inverse=true)
        Sx_, Sy_ = deepcopy(Sx), deepcopy(Sy)
        gate_application!(peps, gate_op, Sx, Sy, su_bdim; kwargs...)
        order == 2 && gate_application!(peps, gate_op, Sx, Sy, su_bdim; reversed=true, kwargs...)
        absorb_S_matrices_into_peps!(peps, Sx, Sy; inverse=false)
        relative_diff = relative_difference(Sx_, Sy_, Sx, Sy)
        verbose && println("Iteration $i relative difference: $relative_diff")
        if relative_diff < convergence_cutoff
            verbose && println("Cutoff reached, stopping simple update at iteration $i")
            break
        end
    end
    peps.double_layer_envs = nothing # the cached environments belong to the old tensors
    return peps
end
simple_update(peps,ham;kwargs...) = simple_update!(deepcopy(peps),ham;kwargs...)

function simple_update_circuit!(peps,gate_op;iter=100,su_bdim=nothing,verbose=true,kwargs...) #Todo: add variable su_bdim
    su_bdim === nothing && (su_bdim = peps.bond_dim)
    check_su_bdim(peps, su_bdim)
    Sx,Sy,_,_,_ = QuantumNaturalfPEPS.super_orthonormalization!(peps; k=100, error=1e-6)
    for i in 1:iter
        absorb_S_matrices_into_peps!(peps, Sx, Sy; inverse=true)
        Sx_, Sy_ = deepcopy(Sx), deepcopy(Sy)
        gate_application!(peps, gate_op, Sx, Sy, su_bdim; kwargs...)
        absorb_S_matrices_into_peps!(peps, Sx, Sy; inverse=false)
    end
    peps.double_layer_envs = nothing # the cached environments belong to the old tensors
    return peps
end
simple_update_circuit(peps,gate_op;kwargs...) = simple_update_circuit!(deepcopy(peps),gate_op;kwargs...)

