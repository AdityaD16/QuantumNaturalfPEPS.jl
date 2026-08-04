# =============================================================================
# Symmetric.jl  --  U(1)-symmetric (QN block-sparse) support for QuantumNaturalfPEPS
# =============================================================================
#
# This file is *additive*. It plugs into the existing PEPS machinery through:
#   - new methods of `permute_(reshape_)and_copy!` for BlockSparse storage
#     (pure multiple-dispatch additions, nothing existing is shadowed), and
#   - three `hasqns(...)` branches in PEPS.jl (`vec`, `write!`, `length`) that
#     delegate here when the PEPS carries quantum numbers.
#
# Scope: ABELIAN symmetries only (U(1) / Z_n), because ITensors' QN system is
# abelian. True SU(2) is NOT supported by this backend; the realistic option is
# to impose the U(1) subgroup (Sz conservation). See `make_u1_peps`.
#

# NDTensors is reached through `using ITensors` (same as tensor_ops.jl); do NOT
# `using NDTensors`, it is not a direct dependency of this package.

# -----------------------------------------------------------------------------
# Per-tensor parameter count
# -----------------------------------------------------------------------------
# Dense: full index-product (matches `length(ITensors.tensor(t))`).
# QN   : only the stored (symmetry-allowed) elements -> nnz.
# This is THE invariant that makes the natural gradient preserve the symmetry:
# θ only ever parametrizes allowed blocks, so forbidden entries can never be
# excited by an update.
tensor_nparams(t::ITensor) = hasqns(t) ? NDTensors.nnz(t) : length(ITensors.tensor(t))

# -----------------------------------------------------------------------------
# BlockSparse kernels  (mirror tensor_ops.jl's DenseTensor methods)
# -----------------------------------------------------------------------------
# For a QN ITensor we work in a *canonical index order* (the caller passes
# `target_indices`, always (siteind, linkinds...) just like the dense path).
# `permute` reorders indices while preserving block structure, and `data`
# returns the contiguous stored elements in a deterministic order for a fixed
# index order -> the flatten/unflatten roundtrip is self-consistent.

function permute_reshape_and_copy!(dest, tensor::NDTensors.BlockSparseTensor, target_indices)
    tp = permute(itensor(tensor), target_indices...)
    dest .= NDTensors.data(ITensors.tensor(tp))
    return dest
end

function permute_and_copy!(dest, tensor::NDTensors.BlockSparseTensor, target_indices)
    tp = permute(itensor(tensor), target_indices...)
    dest .= NDTensors.data(ITensors.tensor(tp))
    return dest
end

# -----------------------------------------------------------------------------
# vec / length / write!  (QN variants, called from the PEPS.jl hooks)
# -----------------------------------------------------------------------------
function length_qn(peps::AbstractPEPS; mask=peps.mask)
    x = 0
    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        mask[i, j] != 0 && (x += tensor_nparams(peps[i, j]))
    end
    return x
end

function vec_qn(peps::AbstractPEPS; mask=peps.mask)
    T = eltype(peps)
    θ = Vector{T}(undef, length_qn(peps; mask))
    pos = 1
    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        mask[i, j] == 0 && continue
        n = tensor_nparams(peps[i, j])
        x = @view θ[pos:pos+n-1]
        permute_reshape_and_copy!(x, peps[i, j], (siteind(peps, i, j), linkinds(peps, i, j)...))
        pos += n
    end
    return θ
end

function write_qn!(peps::AbstractPEPS, θ::Vector{T}; reset_double_layer=true, mask=peps.mask) where {T}
    @assert eltype(peps) == T "PEPS eltype $(eltype(peps)) != θ eltype $T"
    pos = 1
    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        mask[i, j] == 0 && continue
        n = tensor_nparams(peps[i, j])
        θi = @view θ[pos:pos+n-1]
        pos += n

        # Reuse the existing tensor as the structural template (its QN index
        # arrows + allowed blocks are fixed); only overwrite the stored data,
        # in the canonical (siteind, linkinds...) order.
        target = (siteind(peps, i, j), linkinds(peps, i, j)...)
        tp = permute(peps[i, j], target...)
        NDTensors.data(ITensors.tensor(tp)) .= θi
        peps[i, j] = tp
    end
    reset_double_layer && (peps.double_layer_envs = nothing)
    return peps
end

# -----------------------------------------------------------------------------
# QN-aware gradient flatten (get_Ok)
# -----------------------------------------------------------------------------
# The dense `get_Ok` (Ok.jl) writes the per-site gradient directly into a flat
# vector with a `loc_dim` stride, materializing zeros for non-sampled spins.
# That layout assumes dense storage. The QN-clean way is to build the per-site
# gradient as an ITensor with the SAME indices as peps[i,j] and flatten it with
# the very same helper used by `vec_qn` -- so dense and QN share one convention.
#
# grad_tensor(i,j) = ok_tensor(i,j) ⊗ projector(siteind => sampled spin)
#
# Call this from a `hasqns` branch in get_Ok (see the sketch in your notes).
function flatten_site_gradient!(dest_view, peps, ok_tensor, S, i, j)
    t = peps[i, j]
    s = siteind(peps, i, j)

    # grad lives in the DUAL space: ok_tensor's link legs carry arrows opposite
    # to peps[i,j], and the projector selects only the sampled physical sector.
    grad = ok_tensor * get_projector(eltype(peps), S[i, j], s)

    # Bring grad onto peps[i,j]'s primal arrows: dag flips arrows back (and
    # conjugates), conj undoes the conjugation -> holomorphic gradient in the
    # primal index space, matching peps[i,j] exactly.
    grad = conj(dag(grad))

    # Embed into the full block structure: zero(t) carries every allowed block
    # (the non-sampled physical sector stays zero), so nnz matches vec_qn.
    G = zero(t) + grad
    permute_reshape_and_copy!(dest_view, G, (s, linkinds(peps, i, j)...))
    return dest_view
end

# -----------------------------------------------------------------------------
# U(1) initialization  (finishes the scaffold commented at the bottom of PEPS.jl)
# -----------------------------------------------------------------------------
"""
    u1_bond(q_to_dim; tags)

Build a U(1) virtual bond Index from a list of `qn_value => block_dim` pairs,
e.g. `u1_bond([-1=>1, 0=>2, 1=>1]; tags="h,1,2")`.
"""
u1_bond(q_to_dim; tags="Link") =
    Index((QN("Sz", q) => d for (q, d) in q_to_dim)...; tags=tags, dir=ITensors.Out)

"""
    u1_tensor_init(S, ingoing, outgoing; flux=QN("Sz", 0))

Random QN ITensor with the given total `flux`, indices `[ingoing...; outgoing...]`.
The flux fixes the conserved-charge sector of the whole tensor; total Sz of the
PEPS is the sum of per-site fluxes.
"""
function u1_tensor_init(::Type{S}, ingoing, outgoing; flux=QN("Sz", 0)) where {S<:Number}
    is = vcat(ingoing, outgoing)
    return randomITensor(S, flux, is...)
end

"""
    make_u1_peps(S, Lx, Ly; sectors, flux, bond kwargs...)

Construct a U(1) (Sz-conserving) PEPS. `sectors` lists the virtual-bond charge
content as `qn => block_dim` pairs (its sum of dims is the effective bond_dim).
Physical indices are S=1/2 with `conserve_qns=true`.

VALIDATE: this gives a *valid* QN PEPS but NOT an isometric one. The isoPEPS
init path (random_unitary) has no QN analogue here yet; if the natural-gradient
preconditioner relies on near-isometry, add a block-wise QR/SVD gauge after init.
"""
function make_u1_peps(::Type{S}, Lx::Int, Ly::Int;
                      sectors=[-1 => 1, 0 => 2, 1 => 1],
                      site_flux=QN("Sz", 0), kwargs...) where {S<:Number}
    # Build QN-carrying physical indices directly: the package's 3-arg
    # `siteinds(type, Lx, Ly)` override (PEPS.jl) swallows kwargs, so we cannot
    # pass conserve_qns through it.
    hilbert = [siteind("S=1/2"; conserve_qns=true, addtags="nx=$i,ny=$j")
               for i in 1:Lx, j in 1:Ly]
    bond_dim = sum(d for (_, d) in sectors)

    h_links = Matrix{Index}(undef, Lx, Ly - 1)
    v_links = Matrix{Index}(undef, Lx - 1, Ly)
    for i in 1:Lx, j in 1:Ly-1
        h_links[i, j] = u1_bond(sectors; tags="h_link,$i;$j -> $i;$(j+1)")
    end
    for i in 1:Lx-1, j in 1:Ly
        v_links[i, j] = u1_bond(sectors; tags="v_link,$i;$j -> $(i+1);$j")
    end

    tensors = Array{ITensor}(undef, Lx, Ly)
    for i in 1:Lx, j in 1:Ly
        ingoing = Index[hilbert[i, j]]
        outgoing = Index[]
        j != Ly && push!(outgoing, h_links[i, j])      # right
        i != Lx && push!(outgoing, v_links[i, j])      # down
        j != 1  && push!(ingoing, dag(h_links[i, j-1]))   # left  (dag => In)
        i != 1  && push!(ingoing, dag(v_links[i-1, j]))   # up
        tensors[i, j] = u1_tensor_init(S, ingoing, outgoing; flux=site_flux)
    end

    return PEPS(tensors, bond_dim; kwargs...)
end

make_u1_peps(Lx::Int, Ly::Int; kwargs...) = make_u1_peps(Float64, Lx, Ly; kwargs...)
