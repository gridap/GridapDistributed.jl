module HeatEquationTestsSeq
using PartitionedArrays
include("../HeatEquationTests.jl")
with_backend(HeatEquationTests.main,SequentialBackend(),(2,2))
end # module
