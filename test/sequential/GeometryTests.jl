module GeometryTestsSeq
using PartitionedArrays
include("../GeometryTests.jl")
with_debug(GeometryTests.main,(2,2))
with_debug(GeometryTests.main,(2,2,2))
end
