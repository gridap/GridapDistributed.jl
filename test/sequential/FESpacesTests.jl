module FESpacesTestsSeq

using PartitionedArrays

include("../FESpacesTests.jl")

parts = get_part_ids(sequential,(2,2))
FESpacesTests.main(parts)

end # module
