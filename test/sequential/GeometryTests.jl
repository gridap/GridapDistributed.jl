module GeometryTestsSeq
using PartitionedArrays
include("../GeometryTests.jl")
with_backend(GeometryTests.main,SequentialBackend(),(2,2))
with_backend(GeometryTests.main,SequentialBackend(),(2,2,2))
end
