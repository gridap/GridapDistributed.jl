module HeatEquationTestsSeq
using PartitionedArrays
include("../HeatEquationTests.jl")

with_debug() do distribute
    HeatEquationTests.main(distribute,(2,2))
end

end # module
