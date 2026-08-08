function generate_double_layer_env_row(peps_row, sites, maxdim; cutoff=1e-13)
    bra = noprime.(prime.(dag.(peps_row)), sites') # dag (not conj) dualizes QN arrows; == conj for non-QN
    bra = MPO(bra)
    ket = MPO(peps_row)

    E_mpo = contract(bra, ket; maxdim, cutoff)
    E_mps = MPS(E_mpo[:])
    
    return Environment(E_mps; normalize=true)
end

function generate_double_layer_env_row(peps_row, sites, peps_double_env, maxdim; cutoff=1e-13)
    bra = noprime.(prime.(dag.(peps_row)), sites') # dag (not conj) dualizes QN arrows; == conj for non-QN

    E_mpo = MPO(peps_row .* bra)
    E_mps = contract(E_mpo, peps_double_env.env; maxdim, cutoff) # This costs (D^2 * maxdim) ^ 3, expensive!

    return Environment(E_mps, peps_double_env.f; normalize=true)
end

function generate_double_layer_envs(peps::AbstractPEPS)
    Lx = size(peps, 1)
    
    maxdim = peps.double_contract_dim
    cutoff = peps.double_contract_cutoff
    sites = siteinds(peps)

    # for every row we calculate the double layer environment
    double_layer_envs = Vector{Environment}(undef, Lx - 1)
    double_layer_envs[end] = generate_double_layer_env_row(peps[Lx, :], sites[Lx, :], maxdim; cutoff)

    for i in Lx-1:-1:2
        double_layer_envs[i-1] = generate_double_layer_env_row(peps[i, :], sites[i, :], double_layer_envs[i], maxdim; cutoff)
    end
    return double_layer_envs
end

# adds the double layer environments to the PEPS
function update_double_layer_envs!(peps::AbstractPEPS)
    peps.double_layer_envs = generate_double_layer_envs(peps) 
end

###########################################
# Sampling
###########################################


# calculates the the ket layer for the smaplings and applies (if available) already sampled rows (from above)
function get_ket(peps, i, env_top=nothing)
    ket = MPO(peps[i, :])
    if i == 1
        return ket
    end
    @assert env_top != nothing "env_top is not defined"
    return contract(ket, env_top[i-1].env; maxdim=peps.sample_dim, cutoff=peps.sample_cutoff)
end

# calculates the unsampled contractions along a row (from right to left the sites are contracted along the physical Index)
function calculate_unsampled_Env_row(ket, bra, peps, i, sites)
    Ly = size(peps, 2) 
    E = Vector{ITensor}(undef, Ly-1)
    bra = noprime.(bra, sites')

    if i == size(peps, 1)
        E[end] = bra[end] * ket[end]
        for j in Ly-1:-1:2
            E[j-1] = E[j] * ket[j] * bra[j]
        end
        return E
    end

    E[end] = bra[end] * peps.double_layer_envs[i].env[end] * ket[end]
    for j in Ly-1:-1:2
        E[j-1] = E[j] * ket[j] * peps.double_layer_envs[i].env[j] * bra[j]
    end
    return E
end

# returns the phys_dimxphys_sim matrix ρ_r which is needed to sample from. Also updates sigma (used to store the contraction of already sampled sites from the left edge to the current site)
function get_reduced_ρ(ket_j, bra_j, peps, i, j, E, sigma)
   
    if i != size(peps, 1)
        # uncombined_double_layer = peps.double_layer_envs[i].env[j]
        sigma = sigma * ket_j * peps.double_layer_envs[i].env[j] * bra_j
    else
        sigma = sigma * ket_j * bra_j
    end

    if j == size(peps, 2)
        ρ_r = sigma
        return ρ_r, sigma
    end

    ρ_r = sigma * E[j]
    return ρ_r, sigma
end

# samples from ρ_r and updates pc
# `mask[v+1] == true` forbids outcome v: its probability is zeroed BEFORE
# renormalization, so the returned pc is the conditional of the constrained
# proposal distribution p̃c (PRB 104, 235141, Sec. IV). The downstream
# importance weights 2logψ - logpc correct for p̃c ≠ |ψ|² automatically.
function sample_ρr(ρ_r, mask=nothing)
    k = size(ρ_r, 1)
    T = real(eltype(ρ_r))
    p = Vector{T}(undef, k)
    for i in 1:k
        p[i] = abs(ρ_r[i, i])
        im = abs(imag(ρ_r[i, i]))
        # change assert to warning
        if im/(p[i] + 1e-10) >= 1e-6 && im >= 1e-8
            @warn "ρ_r is not real $(ρ_r[i,i])"
        end
        # @assert im/(p[i] + 1e-10) < 1e-6 || im < 1e-12 "ρ_r is not real $(ρ_r[i,i])"
    end
    if mask !== nothing
        @assert length(mask) == k "constraint mask length $(length(mask)) != local dim $k"
        p[mask] .= zero(T)
        sum(p) > 0 || error("sector constraint eliminated all outcomes (state has no weight left in the target sector)")
    end
    i = sample_p(p, normalize=true)
    return i-1, p[i]
end

function sample_p(probs::Vector{T}; normalize=true) where T<:Real
    if normalize
        probs ./= sum(probs)
    end
    r = rand()
    psum = 0
    for (i, p) in enumerate(probs)
        psum += p
        if psum > r
            return i
        end
    end
    isapprox(sum(probs), 1.0, atol=1e-4) || error("probs is not normalized sum(probs)=$(sum(probs))")
    return length(probs)  # fallback to last elementend
end

"""
    sz_sector_counts(peps, Q=0)

Per-species caps enforcing total Sz = `Q` for S=1/2 sites (sampled value 0 = ↑,
1 = ↓): a configuration has Sz = Q iff #↑ = N/2 + Q and #↓ = N/2 - Q.
Pass the result as `max_counts` to `get_sample` / `Ok_and_Ek` /
`generate_Oks_and_Eks` to restrict sampling to that sector
(constrained direct sampling, PRB 104, 235141, Sec. IV).
"""
function sz_sector_counts(peps::AbstractPEPS, Q::Real=0)
    N = prod(size(peps))
    n_up = N / 2 + Q
    isinteger(n_up) && 0 <= n_up <= N ||
        error("total Sz = $Q is not reachable with $N spin-1/2 sites")
    return [Int(n_up), N - Int(n_up)]
end

# Species-count change of one <S|O|S'> term, in the two key formats
# get_precomp_sOψ_elems produces. Only the sites that actually change are listed
# in a flip tuple, so unlisted sites cannot contribute to the delta.
_sector_delta!(Δ, flip_term::Tuple, S) = begin           # ((site, s'_i), ...)
    for (site, val) in flip_term
        Δ[val + 1] += 1
        Δ[S[site...] + 1] -= 1
    end
    Δ
end
_sector_delta!(Δ, S_flipped::AbstractArray, S) = begin   # the flipped sample itself
    for v in S_flipped
        Δ[v + 1] += 1
    end
    for v in S
        Δ[v + 1] -= 1
    end
    Δ
end

"""
    keep_sector_preserving!(Ek_terms, S, max_counts)

Drop the off-diagonal terms `<S|O|S'>` whose flipped configuration S' leaves the
sector that constrained sampling (`max_counts`) restricts S to.

Sector-restricted sampling makes the weighted average over samples
`Σ_{S∈sector} |ψ(S)|² O_loc(S) / Σ_{S∈sector} |ψ(S)|²`. Letting O_loc sum over
every S' turns that into `<ψ|P O|ψ> / <ψ|P|ψ>` -- the projector on the bra side
only, which is not the expectation value of any state and is not even real in
general. Restricting S' to the sector as well gives `<Pψ|O|Pψ> / <Pψ|Pψ>`.

The sector is fixed by the per-species counts, so a term survives iff its flips
leave every count unchanged. Operators that commute with P (the Hamiltonian, any
charge-conserving observable) keep all their terms and are unaffected; operators
that do not (single-site Sx, Sy) correctly collapse to zero.
"""
function keep_sector_preserving!(Ek_terms, S, max_counts)
    Δ = zeros(Int, length(max_counts))
    for key in collect(keys(Ek_terms))
        key === () && continue          # diagonal term, always in the sector
        all(iszero, _sector_delta!(fill!(Δ, 0), key, S)) || delete!(Ek_terms, key)
    end
    return Ek_terms
end

# generates a sample of a given peps along with pc and the top environments
# max_counts: optional per-species caps (see sz_sector_counts); sum(max_counts)
# must equal the number of sites so that hitting all caps == hitting the sector.
function get_sample(peps::AbstractPEPS; mode::Symbol=:full, alg="densitymatrix", timer=TimerOutput(),
                    max_counts=nothing, kwargs...)
    S = Array{Int64}(undef, size(peps))
    local counts
    if max_counts !== nothing
        @assert sum(max_counts) == prod(size(peps)) "sum(max_counts)=$(sum(max_counts)) must equal the number of sites $(prod(size(peps)))"
        counts = zeros(Int, length(max_counts))
    end
    
    env_top = Array{Environment}(undef, size(peps, 1)-1)
    sites = siteinds(peps)
    ρ_r = ITensor()
    
    logpc = 0
    # we loop through every row
    for i in 1:size(peps, 1)
        sigma = 1
        ket = @timeit timer "env_sample" get_ket(peps, i, env_top)
        bra = prime.(dag.(ket[:])) # dag (not conj) dualizes QN arrows; == conj for non-QN

        # we then calculate the unsampled environment (in one row)
        E = @timeit timer "env_row" calculate_unsampled_Env_row(ket, bra, peps, i, sites[i, :])

        # then we loop through the different sites in one row
        for j in 1:size(peps, 2)
            
            # calculate the phys_dimxphys_dim matrix from which we sample
            ρ_r, sigma = get_reduced_ρ(ket[j], bra[j], peps, i, j, E, sigma)
            
            # sample from ρ_r (masking species that hit their sector cap)
            if max_counts === nothing
                S[i, j], pc = sample_ρr(ρ_r)
            else
                S[i, j], pc = sample_ρr(ρ_r, counts .>= max_counts)
                counts[S[i, j] + 1] += 1
            end
            logpc += log(pc)
            
            # after the sampling of the current site, it is fixed and its contraction with the aleady sampled sites is stored in sigma
            site = siteind(peps, i, j)
            # ket leg (<Out>) wants the dag'd projector (get_projector dags internally);
            # bra leg is dag'd (<In>), so double-dag its index to restore the native arrow.
            sigma = sigma * get_projector(S[i, j], sites[i, j]) * get_projector(S[i, j], dag(sites[i, j]'))
            sigma ./= pc # we divide by pc to avoid numerical issues
        end
        
        if mode === :fast
            # the sampled bra is used to generate the top environments
            ket = ket .* [get_projector(S[i, j], siteind(peps, i, j)) for j in 1:size(peps, 2)]
            if i == 1
                env_top[i] = Environment(MPS(ket.data); normalize=true)
            elseif i != size(peps, 1) 
                env_top[i] = Environment(MPS(ket.data), env_top[i-1].f; normalize=true)
            end

        elseif mode === :full
             # Should we be recalculating the top environment here? Is it slower?
             # The answer is yes, it is slower, but not by match. But it is also more accurate.
            if i == 1
                peps_projected_1 = get_projected(peps, S, 1, :)
                @timeit timer "env_top" env_top[1] = generate_env_row(peps_projected_1, peps.contract_dim; alg, cutoff=peps.contract_cutoff, kwargs...)
            elseif i != size(peps, 1) 
                peps_projected_row = get_projected(peps, S, i, :)
                @timeit timer "env_top" env_top[i] = generate_env_row(peps_projected_row, peps.contract_dim; env_row_above=env_top[i-1], alg, cutoff=peps.contract_cutoff, kwargs...)
            end  
        end
    end
    
    return S, logpc, env_top
end