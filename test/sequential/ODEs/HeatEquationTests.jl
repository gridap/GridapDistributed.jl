module HeatEquationTestsSeq
using PartitionedArrays
include("../../ODEs/HeatEquationTests.jl")
prun(HeatEquationTests.main,sequential,(2,2))
end # module
