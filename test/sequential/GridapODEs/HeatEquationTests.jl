module HeatEquationTestsSeq
using PartitionedArrays
include("../../GridapODEs/HeatEquationTests.jl")
prun(HeatEquationTests.main,sequential,(2,2))
end # module
