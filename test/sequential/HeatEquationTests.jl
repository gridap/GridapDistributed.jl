module HeatEquationTestsSeq
using PartitionedArrays
include("../HeatEquationTests.jl")
prun(HeatEquationTests.main,sequential,(2,2))
end # module
