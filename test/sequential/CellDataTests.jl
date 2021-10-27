module CellDataTestsSeq

using PartitionedArrays

include("../CellDataTests.jl")

parts = get_part_ids(sequential,(2,2))
CellDataTests.main(parts)

end # module
