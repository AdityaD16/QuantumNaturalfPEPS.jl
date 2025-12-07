using LinearAlgebra
using Statistics
using JLD2
using ITensors
using QuantumNaturalGradient
using QuantumNaturalfPEPS
using Distributed
using TimerOutputs
using Random
using Revise


function callback(;state,natural_gradient,theta_dot, niter=1, kwargs...)
    kwargs = Dict(string(key) => v for (key, v) in kwargs)
    kwargs["params"] = params
    kwargs["solver"] = solver
    kwargs["integrator"] = integrator
    kwargs["peps"] = peps
    kwargs["timer"] = timer

    push!(energy_arr,state.energy)
    # energy_change = state.energy - prev_energy[]

    # if energy_change < 0# Big improvement
    # #     # Keep LR the same or increase it slightly
    #     # state.integrator.lr = min(1, 1.05 * state.integrator.lr)
    #     # println("Significant energy decrease, increasing learning rate to ", state.integrator.lr)
    # end
    # if energy_change >= 0 # Energy increased or stalled (bad step)
    #     # Definitely decrease the learning rate
    #     # state.integrator.lr = max(0.8 * state.integrator.lr, 0.00001)
    #     # println("Energy did not decrease, decreasing learning rate to ", state.integrator.lr)
 
    #     count[] += 1
    # end
    # prev_energy[] = state.energy

    # if count[]>=10
    #     state.sample_nr = min(state.sample_nr + 250,3000)
    #     # state.integrator.lr = 0.1
    #     println("Increasing sample number to ", state.sample_nr)
    #     # println("Setting learning rate to ", state.integrator.lr)
    #     count[] = 0
    # end

    # println("Iteration $niter: Energy Density = ", state.energy / (params[:Lx]*params[:Ly]))
        # throw(ErrorException("stop"))
    if state.energy < best_energy[]
        best_energy[] = state.energy
        best_theta[] = deepcopy(state.θ)
        println("Saving best PEPS with energy ", best_energy[])
        # JLD2.save("DDL4.jld2", "peps",peps)
        count_best[] = 0
    end
    # print("Importance weights stats: ", natural_gradient.importance_weights, "\n")
    # JLD2.save("SS1_thetadot.jld2", "theta_dot",theta_dot)
    # if niter == 1
    #     throw(ErrorException("stop"))
    # end

    count_best[] += 1
    JLD2.save("DD:L$(params[:Lx]),D$(params[:bdim]),chi$(params[:double_contract_dim]).jld2", "peps",peps)

    # if niter>20 && count_best[]%25 == 0
    #     println("No improvement for 25 iterations, best PEPS with energy ", best_energy[])
    #     state.integrator.lr = max(state.integrator.lr /2)
    #     println("Decreasing learning rate to ", state.integrator.lr)
    # end
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

    # if count_best[]%100 ==0
    #     state.integrator.lr = 0.001
    #     println("Decreasing learning rate to ", state.integrator.lr)
    # end

    if count_best[]==300
        println("No improvement for 300 iterations, stopping optimization.")
        return false
    end
    # if niter == 300
    #     println("Reached maximum number of iterations, stopping optimization.")
    #     return false
    # end 

    # JLD2.save("SSsave.jld2", "peps",peps)
end

# g_SS = JLD2.load("SS_thetadot.jld2", "theta_dot")
# norm_g_SS = norm(g_SS)
# println("Norm of SS theta_dot = $norm_g_SS")
# g_SS1 = JLD2.load("SS1_thetadot.jld2", "theta_dot")
# norm_g_SS1 = norm(g_SS1)
# # println("Norm of SS1 theta_dot = $norm_g_SS1")
# g_DD = JLD2.load("DD_thetadot.jld2", "theta_dot")
# norm_g_DD = norm(g_DD)
# println("Norm of DD theta_dot = $norm_g_DD")
# # # # # # g_SD = JLD2.load("SD_thetadot.jld2", "theta_dot")
# # # # # # norm_g_SD = norm(g_SD)
# # # # # # println("Norm of SD theta_dot = $norm_g_SD")
# g_ = norm(g_DD - g_SS)/norm(g_SS)
# println("Norm difference between SS and DD theta_dot = $g_")
# # # # g_ = norm(g_DD - g_SD)
# # # # println("Norm difference between SD and DD theta_dot = $g_")
# # # g_ = norm(g_SS1 - g_DD)/norm(g_SS1)
# # # println("Norm difference between SS1 and SD theta_dot = $g_")
# g_ = norm(g_SS1 - g_SS)/norm(g_SS)
# println("Norm difference between SS1 and SS theta_dot = $g_")
# throw(ErrorException("stop"))

# mpi = true
 threaded = true
 multiproc = true 
# println("Starting XXX PEPS simulation with multithreading = $threaded and multiproc = $multiproc")

# # Initialize 2 processes with 8 threads each, assuming you have a 16 core cpu
 nr_procs, nr_threads = 2,4
# addprocs(nr_procs, exeflags="-t $nr_threads")

println("Number of processes: ", nprocs())
println("Number of threads per process: ", Threads.nthreads())

@everywhere begin
    using QuantumNaturalfPEPS
    using LinearAlgebra
    BLAS.set_num_threads(1) # Each thread should only use one Blas thread
end
BLAS.set_num_threads(8) # For computing the double layers
# # # ctheta = 2/3 +(1/3)*1

# Simulation Parameters
params = Dict(
    :T              => ComplexF64,
    :seed          => 20,
    :Lx             => 3,
    :Ly             => 3,
    :bdim           => 3,
    :lr             =>0.05,
    :J             => -1,
    :h             => 3.0,
    :eigencut       => 1e-7,
    :contract_cutoff=> 1e-10,
    :sample_cutoff  => 1e-7,
    :contract_dim   => 500,
    :maxiter        =>500,
    :α_init         => 2.0,
    :sample_nr      => 1000,
    :double_contract_dim => 2,
    :double_contract_cutoff => 1e-20,
)

# peps.contract_cutoff = params[:contract_cutoff]
println("Simulation parameters: ")
for (k,v) in params
    println("$k => $v")
end

Random.seed!(params[:seed])
# Function to compute the structure factor S(q) for given q
function structure_factor(q::Tuple{Any,Any}, coords::Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}, corrmat::Matrix{Float64}, params::Dict)
    S_q = 0.0
    N = params[:Lx] * params[:Ly]
    for i in 1:params[:Lx], j in 1:params[:Ly]
        (xi, yi) = coords[(i,j)]
        idx = (i-1)*params[:Ly] + j
        for k in 1:params[:Lx], l in 1:params[:Ly]
            (xk, yk) = coords[(k,l)]
            idx2 = (k-1)*params[:Ly] + l
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
Random.seed!(params[:seed])
# Define the Lattice and PEPS
hilbert = siteinds("S=1/2", params[:Lx], params[:Ly])
peps = PEPS(params[:T], hilbert; bond_dim=params[:bdim], show_warning=false,
    contract_cutoff=params[:contract_cutoff], 
    contract_dim=params[:contract_dim], 
    sample_cutoff=params[:sample_cutoff],
)

# # Multiply the spectrum of the PEPS by a power-law factor to make it contractible
QuantumNaturalfPEPS.multiply_algebraic_spectrum!(peps, params[:α_init])
# peps = JLD2.load("DDL$(params[:Lx]).jld2", "peps")


# println("After super-orthonormalization: res = $res, lognorm = $lognorm")

# print(ITensors.siteind(peps,1,3))

ham_J1 = OpSum()


a = 1.0 #lattice constant
coords = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
for i in 1:params[:Lx], j in 1:params[:Ly]
     y = (j-1) 
    x = (i-1) 
    coords[(i,j)] = (x,y)
    # println("Coordinate of site ($i,$j) = ($(x),$(y))")
end


# Define the magnetic moment direction
for i in 1:params[:Lx], j in 1:params[:Ly]
    for k in 1:params[:Lx], l in 1:params[:Ly]
        # skip self-interaction
        if i == k && j == l
            continue
        end
        # avoid double-counting
        if (k < i) || (k == i && l < j)
            xi,yi = coords[(i,j)]
            xk,yk = coords[(k,l)]
            rij = [xi - xk, yi - yk]
            # println("Coordinates of site ($i,$j): ($xi,$yi), site ($k,$l): ($xk,$yk)")
            r_ij = sqrt((xi - xk)^2 + (yi - yk)^2)
            # println("Distance between sites ($i,$j) and ($k,$l) = $r_ij")
            if !(isapprox(r_ij, 1.0; atol=1e-5))
                continue
            end
            # println("Adding interaction between sites ($i,$j) and ($k,$l)")
            interaction = params[:J]
            ham_J1 .+= (interaction, "Sz", (i,j), "Sz", (k,l))

        end
    end
end

function construct_gates(ham_op)
    gate_op = ham_op
    for n in 1:length(length(gate_op.tensors))
        gate_op.tensors[n] = exp(-dt * gate_op.tensors[n])
    end
    return gate_op


    

function horizontal_gate_application!(peps,gate,site1,site2,Sx,Sy)

    x1,y1 = site1
    x2,y2 


if x1 == 1 
                    down_bond1 = commonind(peps[x1,y1], peps[x1+1,y1])
                    down_bond2 = commonind(peps[x2,y2], peps[x2+1,y2])
                    _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sx[x1,y1,:],down_bond1)
                    _,s_invs_2 = QuantumNaturalfPEPS.Smatrix(Sx[x2,y2,:],down_bond2)
                    u = apply(u,s_inv1)
                    v = apply(v,s_invs_2)
                elseif x1 == Nx
                    up_bond1 = commonind(peps[x1-1,y1], peps[x1,y1])
                    up_bond2 = commonind(peps[x2-1,y2], peps[x2,y2])
                    _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sx[x1-1,y1,:],up_bond1)
                    _,s_invs_2 = QuantumNaturalfPEPS.Smatrix(Sx[x1-1,y2,:],up_bond2)
                    u = apply(u,s_inv1)
                    v = apply(v,s_invs_2)
                else
                    up_bond1 = commonind(peps[x1-1,y1], peps[x1,y1])
                    down_bond1 = commonind(peps[x1,y1], peps[x1+1,y1])
                    up_bond2 = commonind(peps[x2-1,y2], peps[x2,y2])
                    down_bond2 = commonind(peps[x2,y2], peps[x2+1,y2])
                    # println(eltype(Sx[x1-1,y1,:]))
                    _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sx[x1-1,y1,:],up_bond1)
                    # println(eltype(s_inv1))
                    _,s_inv2 = QuantumNaturalfPEPS.Smatrix(Sx[x1,y1,:],down_bond1)
                    _,s_invs_3 = QuantumNaturalfPEPS.Smatrix(Sx[x2-1,y2,:],up_bond2)
                    _,s_invs_4 = QuantumNaturalfPEPS.Smatrix(Sx[x2,y2,:],down_bond2)
                    # # println(u)
                    # println(eltype(s_inv1))
                    # # s_inv1 = convert(ITensor{Float64}, s_inv)
                    # s_inv1 = replace_eltype()
                    println(Sx[x1-1,y1,:])
                    println(s_inv1)
                    u = apply(u,s_inv1)
                    u = apply(u,s_inv2)
                    v = apply(v,s_invs_3)
                    v = apply(v,s_invs_4)
                end





function gate_application(peps,gate_op,Sx,Sy)

    Nx = size(gate_op)[1]
    Ny = size(gate_op)[2]

    coord_to_2d(i, Nx) = begin
        y = div(i-1, Nx) + 1     # row index
        x = mod(i-1, Nx) + 1     # column index
        return x, y
    end

    for j in 1:length(gate_op.tensors)

        z1, z2  = gate_op.sites[j]
        x1, y1 = coord_to_2d(z1, Nx)
        x2, y2 = coord_to_2d(z2, Nx)
        println("Interaction between $x1,$y1 and $x2,$y2")
        @assert abs(x1-x2) + abs(y1-y2) == 1 "Supports only nearest-neighbor interaction on square lattice!"

        if x1==x2
            peps,Sx,Sy = horizontal_gate_application(peps,site1,site2,gate_op.tensor[i],Sx,Sy)
        else
            peps,Sx,Sy = vertical_gate_application(peps,site1,site2,gate_op.tensor[i],Sx,Sy)



        


    





function simple_update(peps,ham,iter=100,dt=1e-3,su_bdim=3)

    gate_op = QuantumNaturalGradient.TensorOperatorSum(ham, peps.hilbert)
    Sx,Sy,peps,_,_ = QuantumNaturalfPEPS.super_orthonormalization(peps; k=1000, error=1e-8, verbose=false)
    gate_op = construct_gates(ham_op)

    for i in 1:iter
        println("Iteration $i")
        for j in 1:length(gate_op.tensors)

            z1, z2  = gate_op.sites[j]
            x1, y1 = coord_to_2d(z1, Nx)
            x2, y2 = coord_to_2d(z2, Nx)
            println("Interaction between $x1,$y1 and $x2,$y2")
            @assert abs(x1-x2) + abs(y1-y2) == 1 "Supports only nearest-neighbor interaction on square lattice!"

            l = commonind(peps[x1,y1], peps[x2,y2])
            if x1==x2
                bi,bj = x1,min(y1,y2) # bond index between the two sites
                S = itensor(diagm((Sy[bi,bj,:])), l, l')
            else
                bi,bj = min(x1,x2),y1 # bond index between the two sites
                S = itensor(diagm((Sx[bi,bj,:])), l, l')
            end
            
            A = apply(peps[x1,y1], S)
            A = apply(A,peps[x2,y2])

            left_indices = inds(peps[x1,y1])
            left_indices = setdiff(left_indices, (l,))

            s1 = siteind(peps, x1, y1)
            s2 = siteind(peps, x2, y2)

            if length(commoninds(A,gate_op.tensors[j])) != 2
                @assert "Error: common indices between A and gate tensor is not 2!"
            end

            A = apply(A, gate_op.tensors[j])

            u, s, v = svd(A,left_indices;maxdim=D)
            bL = inds(s)[1]
            bR = inds(s)[2]
            u = replaceind(u, bL,l)
            v = replaceind(v, bR,l)
            # println(s)
            
            
            # println(eltype(s))

            if x1==x2
                # print(eltype(s))

                Sy[bi,bj,:] = diag(s)
                if x1 == 1 
                    down_bond1 = commonind(peps[x1,y1], peps[x1+1,y1])
                    down_bond2 = commonind(peps[x2,y2], peps[x2+1,y2])
                    _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sx[x1,y1,:],down_bond1)
                    _,s_invs_2 = QuantumNaturalfPEPS.Smatrix(Sx[x2,y2,:],down_bond2)
                    u = apply(u,s_inv1)
                    v = apply(v,s_invs_2)
                elseif x1 == Nx
                    up_bond1 = commonind(peps[x1-1,y1], peps[x1,y1])
                    up_bond2 = commonind(peps[x2-1,y2], peps[x2,y2])
                    _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sx[x1-1,y1,:],up_bond1)
                    _,s_invs_2 = QuantumNaturalfPEPS.Smatrix(Sx[x1-1,y2,:],up_bond2)
                    u = apply(u,s_inv1)
                    v = apply(v,s_invs_2)
                else
                    up_bond1 = commonind(peps[x1-1,y1], peps[x1,y1])
                    down_bond1 = commonind(peps[x1,y1], peps[x1+1,y1])
                    up_bond2 = commonind(peps[x2-1,y2], peps[x2,y2])
                    down_bond2 = commonind(peps[x2,y2], peps[x2+1,y2])
                    # println(eltype(Sx[x1-1,y1,:]))
                    _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sx[x1-1,y1,:],up_bond1)
                    # println(eltype(s_inv1))
                    _,s_inv2 = QuantumNaturalfPEPS.Smatrix(Sx[x1,y1,:],down_bond1)
                    _,s_invs_3 = QuantumNaturalfPEPS.Smatrix(Sx[x2-1,y2,:],up_bond2)
                    _,s_invs_4 = QuantumNaturalfPEPS.Smatrix(Sx[x2,y2,:],down_bond2)
                    # # println(u)
                    # println(eltype(s_inv1))
                    # # s_inv1 = convert(ITensor{Float64}, s_inv)
                    # s_inv1 = replace_eltype()
                    println(Sx[x1-1,y1,:])
                    println(s_inv1)
                    u = apply(u,s_inv1)
                    u = apply(u,s_inv2)
                    v = apply(v,s_invs_3)
                    v = apply(v,s_invs_4)
                end

                if y1+1 == y2
                    if y1!=1
                        left_bond = commonind(peps[x1,y1-1], peps[x1,y1])
                        _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sy[x1,y1-1,:],left_bond)
                        u = apply(u,s_inv1)
                    end
                    if y2 !=Ny
                        right_bond = commonind(peps[x2,y2], peps[x2,y2+1])
                        _,s_invs_2 = QuantumNaturalfPEPS.Smatrix(Sy[x2,y2,:],right_bond2)
                        v = apply(v,s_invs_2)
                    end
                else
                    if y2!=1
                        left_bond = commonind(peps[x2,y2-1], peps[x2,y2])
                        _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sy[x2,y2-1,:],left_bond)
                        v = apply(v,s_inv1)
                    end
                    if y1 !=Ny
                        right_bond = commonind(peps[x1,y1], peps[x1,y1+1])
                        _,s_invs_2 = QuantumNaturalfPEPS.Smatrix(Sy[x1,y1,:],right_bond)
                        u = apply(u,s_invs_2)
                    end
                end
                # print(inds(u), "\n")
                # println(inds(peps[x1,y1]))
                peps[x1,y1] = u
                # println(inds(peps[x1,y1]))

                # break
                peps[x2,y2] = v
                # break
            else
                Sx[bi,bj,:] = diag(s)
                if y1 == 1 
                    right_bond1 = commonind(peps[x1,y1], peps[x1,y1+1])
                    right_bond2 = commonind(peps[x2,y2], peps[x2,y2+1])
                    _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sy[x1,y1,:],right_bond1)
                    _,s_invs_2 = QuantumNaturalfPEPS.Smatrix(Sy[x2,y2,:],right_bond2)
                    u = apply(u,s_inv1)
                    v = apply(v,s_invs_2)
                elseif y1 == Ny
                    left_bond1 = commonind(peps[x1,y1-1], peps[x1,y1])
                    left_bond2 = commonind(peps[x2,y2-1], peps[x2,y2])
                    _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sy[x1,y1-1,:],left_bond1)
                    _,s_invs_2 = QuantumNaturalfPEPS.Smatrix(Sy[x2,y2-1,:],left_bond2)
                    u = apply(u,s_inv1)
                    v = apply(v,s_invs_2)
                else
                    left_bond1 = commonind(peps[x1,y1-1], peps[x1,y1])
                    right_bond1 = commonind(peps[x1,y1], peps[x1,y1+1])
                    left_bond2 = commonind(peps[x2,y2-1], peps[x2,y2])
                    right_bond2 = commonind(peps[x2,y2], peps[x2,y2+1])

                    _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sy[x1,y1-1,:],left_bond1)
                    _,s_inv2 = QuantumNaturalfPEPS.Smatrix(Sy[x1,y1,:],right_bond1)
                    _,s_invs_3 = QuantumNaturalfPEPS.Smatrix(Sy[x2,y2-1,:],left_bond2)
                    _,s_invs_4 = QuantumNaturalfPEPS.Smatrix(Sy[x2,y2,:],right_bond2)
                    u = apply(u,s_inv1)
                    u = apply(u,s_inv2)
                    v = apply(v,s_invs_3)
                    v = apply(v,s_invs_4)   
                end

                if x1+1 == x2
                    if x1!=1
                        up_bond = commonind(peps[x1-1,y1], peps[x1,y1])
                        _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sx[x1-1,y1,:],up_bond)
                        u = apply(u,s_inv1)
                    end
                    if x2 !=Nx
                        down_bond = commonind(peps[x2,y2], peps[x2+1,y2])
                        _,s_invs_2 = QuantumNaturalfPEPS.Smatrix(Sx[x2,y2,:],down_bond)
                        v = apply(v,s_invs_2)
                    end
                else
                    if x2!=1
                        up_bond = commonind(peps[x2-1,y2], peps[x2,y2])
                        _,s_inv1 = QuantumNaturalfPEPS.Smatrix(Sx[x2-1,y2,:],up_bond)
                        v = apply(v,s_inv1)
                    end
                    if x1 !=Nx
                        down_bond = commonind(peps[x1,y1], peps[x1+1,y1])
                        _,s_invs_2 = QuantumNaturalfPEPS.Smatrix(Sx[x1,y1,:],down_bond)
                        u = apply(u,s_invs_2)
                    end
                end
                peps[x1,y1] = u
                peps[x2,y2] = v
            end
        end

    end

# print(peps[2,2])

# siteind(peps,1,1)
# for i in 1:iter
#     println("Iteration $i")

# l = commonind(peps[1,2], peps[2,2])  # returns (dim=2|id=770,"v_link")
# A = peps[2,2] * prime(peps[1,2], l)  # prime levels aligned → contracts along vertical bond
# println(ndims(A))

# end
timer = TimerOutput()


Oks_and_Eks = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, ham_J1; threaded=threaded, multiproc=multiproc, timer) 
# Oks_and_Eks2 = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, ham_J1x; threaded=threaded, multiproc=multiproc, timer)

# Oks_and_Eks2 = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, ham_J1x; threaded=threaded, multiproc=multiproc, timer,peps_preconditioner = p -> basis_change!(p; gate="H"),peps_postconditioner = p -> basis_change!(p; gate="H"))
# Oks_and_Eks3 = QuantumNaturalfPEPS.generate_Oks_and_Eks(peps, ham_J1y; threaded=threaded, multiproc=multiproc, timer,peps_preconditioner = p -> basis_change!(p; gate="ZtoY"),peps_postconditioner = p -> basis_change!(p; gate="YtoZ"))

# Setup the Integrator and Solver
integrator = QuantumNaturalGradient.Euler(lr=params[:lr], write_func=QuantumNaturalfPEPS.write!, 
    basis_func=QuantumNaturalfPEPS.basis_change!, 
    vec=QuantumNaturalfPEPS.vec,gates = ("Id","H","YtoZ"),
    imag = true
    )
solver = QuantumNaturalGradient.EigenSolver(params[:eigencut], verbose=false)

# Initialize Parameters and Evolve
θ = QuantumNaturalGradient.Parameters(peps)

# Logger functions, will be called after each iteration and results will be saved in the misc["history"] dictionary
contract_dim(; contract_dims) = mean(contract_dims)
logger_funcs = [contract_dim]

# Evolve for a fixed (small) number of iterations as a demo
@time loss_value, trained_θ, misc = QuantumNaturalGradient.evolve((Oks_and_Eks), θ; 
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


