module SequentialTests

using Test

@time @testset "DiscreteModels" begin include("DiscreteModelsTests.jl") end

end # module
