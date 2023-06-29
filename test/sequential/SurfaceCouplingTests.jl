module SurfaceCouplingTestsSeq
using PartitionedArrays
include("../SurfaceCouplingTests.jl")
with_debug(SurfaceCouplingTests.main,(2,2))
end # module

