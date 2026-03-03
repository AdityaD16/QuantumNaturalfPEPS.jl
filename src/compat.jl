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

function __init__()
    _patch_itensors_generic_tagset()
    return nothing
end