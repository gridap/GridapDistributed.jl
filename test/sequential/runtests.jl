module SequentialTests

using Test

@time @testset "DistributedDiscreteModel" begin include("DistributedDiscreteModelsTests.jl") end

end # module
