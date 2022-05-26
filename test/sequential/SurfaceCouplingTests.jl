module SurfaceCouplingTestsSeq
using PartitionedArrays
include("../SurfaceCouplingTests.jl")
prun(SurfaceCouplingTests.main,sequential,(2,2))
end # module

