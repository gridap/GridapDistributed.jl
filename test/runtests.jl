module GridapDistributedTests

using Test

@time @testset "DistributedData" begin include("DistributedDataTests.jl") end

@time @testset "DistributedIndexSets" begin include("DistributedIndexSetsTests.jl") end

@time @testset "DistributedVectors" begin include("DistributedVectorsTests.jl") end

@time @testset "CartesianDiscreteModels" begin include("CartesianDiscreteModelsTests.jl") end

@time @testset "DistributedFESpaces" begin include("DistributedFESpacesTests.jl") end

@time @testset "DistributedAssemblers" begin include("DistributedAssemblersTests.jl") end

@time @testset "DistributedPoisson" begin include("DistributedPoissonTests.jl") end

end # module
