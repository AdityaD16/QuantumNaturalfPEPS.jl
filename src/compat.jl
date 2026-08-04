# Fix compability issues with different versions of ITensors.jl, specifically regarding the definition of GenericTagSet. This is needed for JLD2 compatibility, which is used for saving and loading PEPS.
function _patch_itensors_generic_tagset()
    if !isdefined(ITensors, :GenericTagSet)
        if isdefined(ITensors, :TagSets) && isdefined(ITensors.TagSets, :GenericTagSet)
            Core.eval(ITensors, :(const GenericTagSet = TagSets.GenericTagSet))
        elseif isdefined(ITensors, :TagSet)
            Core.eval(ITensors, :(const GenericTagSet = TagSet))
        end
    end
end

# function __init__()
#     # _patch_itensors_generic_tagset()
#     # Quad-layer contractions for interior middle-row sites (Lx≥3) produce
#     # 14-index intermediates as a structural lower bound: h_l(0,1,2,3) +
#     # h_r(0,1,2,3) + v_up(0,1,2,3) + 2 MPS bonds = 14.  Raise the threshold
#     # to 15 so the warning still fires for genuinely unexpected cases.
#     ITensors.set_warn_order(15)
#     return nothing
# end