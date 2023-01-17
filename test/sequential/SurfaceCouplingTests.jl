module SurfaceCouplingTestsSeq
using PartitionedArrays
include("../SurfaceCouplingTests.jl")
with_backend(SurfaceCouplingTests.main,SequentialBackend(),(2,2))
end # module

