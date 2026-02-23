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
            if abs(x1 - x2) + abs(y1 - y2) == 2
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


@inline function two_qubit_gate_application!(peps, site1, site2, gate, Sx, Sy, su_bdim;reduced=true)

    x1,y1 = site1
    x2,y2 = site2
    @assert abs(x1 - x2) + abs(y1 - y2) <= 2 "Supports only nearest-neighbor interaction on square lattice!"
    if abs(x1 - x2) + abs(y1 - y2) == 2
         diagonal_gate_application!(peps, site1, site2, gate, Sx, Sy, su_bdim)
    
    elseif reduced
         reduced_nn_gate_application!(peps, site1, site2, gate, Sx, Sy, su_bdim)
    else
        full_nn_gate_application!(peps, site1, site2, gate, Sx, Sy, su_bdim)
    end
    return peps, Sx, Sy
end
two_qubit_gate_application(peps, site1, site2, gate, Sx, Sy, su_bdim;reduced=true) = two_qubit_gate_application!(deepcopy(peps), site1, site2, gate, deepcopy(Sx), deepcopy(Sy), su_bdim; reduced=reduced)


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
    A = apply(A, gate)

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
    A = apply(A, gate)
    
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
    A = apply(A, gate)

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


@inline function diagonal_gate_application!(peps,site1,site2,gate,Sx,Sy,su_bdim)

    x1,y1 = site1
    x2,y2 = site2

    @assert abs(x1 - x2) + abs(y1 - y2) == 2 "Supports only diagonal interaction on square lattice!"
    # Find the third site involved in the diagonal interaction
    x3,y3 = x1,y2
    x4,y4 = x2,y1
    peps, Sx, Sy = diagonal_gate_application!(peps, (x1,y1),(x2,y2),(x3,y3), gate, Sx, Sy, su_bdim)
    peps, Sx, Sy = diagonal_gate_application!(peps, (x1,y1),(x2,y2),(x4,y4), gate, Sx, Sy, su_bdim)     
    return peps, Sx, Sy
    
end





function gate_application!(peps,gate_op,Sx,Sy,su_bdim;kwargs...)
    Nx = size(gate_op)[1]
    Ny = size(gate_op)[2]

    coord_to_2d(i, Nx) = begin
        y = div(i-1, Nx) + 1     # row index
        x = mod(i-1, Nx) + 1     # column index
        return x, y
    end

    for j in 1:length(gate_op.tensors)
        # Single-site gate
        if length((gate_op.sites[j])) ==1
            z = gate_op.sites[j][1]
            x, y = coord_to_2d(z, Nx)
            peps[x,y] = apply(peps[x,y], gate_op.tensors[j])
            continue
        end
        # Two-site gate
        z1, z2  = gate_op.sites[j]
        x1, y1 = coord_to_2d(z1, Nx)
        x2, y2 = coord_to_2d(z2, Nx)
        # @assert abs(x1-x2) + abs(y1-y2) == 1 "Supports only nearest-neighbor interaction on square lattice!"
        two_qubit_gate_application!(peps, (x1,y1), (x2,y2), gate_op.tensors[j], Sx, Sy, su_bdim; kwargs...)
    end

    return peps, Sx, Sy
end
gate_application(peps,gate_op,Sx,Sy,su_bdim) = gate_application!(deepcopy(peps),gate_op,deepcopy(Sx),deepcopy(Sy),su_bdim)

        
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

@inline function relative_difference( Sx_,Sy_,Sx,Sy)
    relative_diff = 0.0
    for i in 1:length(Sx_)
        s1 = Sx_[i]
        s2 = Sx[i]
        relative_diff += norm(s2-s1) / norm(s1)
    end

    for i in 1:length(Sy_)
        s1 = Sy_[i]
        s2 = Sy[i]
        relative_diff += norm(s2-s1) / norm(s1)
    end

    return relative_diff/(length(Sx_) + length(Sy_))
end


function simple_update!(peps,ham;iter=100,dt=0.01,su_bdim=nothing,convergence_cutoff = 1e-11,kwargs...) #Todo: add variable su_bdim, verbose
    su_bdim === nothing && (su_bdim = peps.bond_dim)
    ham_op = QuantumNaturalGradient.TensorOperatorSum(ham, siteinds(peps))
    gate_op = construct_evolution_gates((ham_op),dt)
    Sx,Sy,_,_,_ = QuantumNaturalfPEPS.super_orthonormalization!(peps; k=100, error=1e-6)
    for i in 1:iter
        absorb_S_matrices_into_peps!(peps, Sx, Sy; inverse=true)
        Sx_, Sy_ = deepcopy(Sx), deepcopy(Sy)
        gate_application!(peps, gate_op, Sx, Sy, su_bdim; kwargs...)
        absorb_S_matrices_into_peps!(peps, Sx, Sy; inverse=false)
        relative_diff = relative_difference(Sx_, Sy_, Sx, Sy)
        println("Iteration $i relative difference: $relative_diff")
        if relative_diff < convergence_cutoff
            println("Cutoff reached, stopping simple update at iteration $i")
            break
        end
    end
    return peps
end
simple_update(peps,ham;iter=10000,dt=0.01,su_bdim=nothing,convergence_cutoff = 1e-11,verbose=true,kwargs...) = simple_update!(deepcopy(peps),ham;iter=iter,dt=dt,su_bdim=su_bdim,convergence_cutoff=convergence_cutoff,verbose=verbose, kwargs...)

