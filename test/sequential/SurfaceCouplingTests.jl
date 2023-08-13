module SurfaceCouplingTestsSeq
using PartitionedArrays
include("../SurfaceCouplingTests.jl")
with_debug() do distribute
    SurfaceCouplingTests.main(distribute,(2,2))
end
end # module

