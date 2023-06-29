module HeatEquationTestsSeq
using PartitionedArrays
include("../HeatEquationTests.jl")
with_debug(HeatEquationTests.main,(2,2))
end # module
