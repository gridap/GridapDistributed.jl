module GeometryTestsSeq
using PartitionedArrays
include("../GeometryTests.jl")
prun(GeometryTests.main,sequential,(2,2))
prun(GeometryTests.main,sequential,(2,2,2))
end
