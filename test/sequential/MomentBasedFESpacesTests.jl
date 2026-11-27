module MomentBasedFESpacesTestsSeq
using PartitionedArrays

include("../MomentBasedFESpacesTests.jl")

# `(2,2,1)` rather than a slab partition: it's the cheapest partition whose
# ranks disagree on the order of a cell's vertex ids.
with_debug() do distribute
  MomentBasedFESpacesTests.main(distribute,(2,2,1))
end

end # module
