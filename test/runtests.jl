module GridapDistributedTests

using Test
using MPI
using GridapDistributedPETScWrappers

@time @testset "DistributedData" begin include("DistributedDataTests.jl") end

@time @testset "DistributedIndexSets" begin include("DistributedIndexSetsTests.jl") end

@time @testset "DistributedVectors" begin include("DistributedVectorsTests.jl") end

@time @testset "CartesianDiscreteModels" begin include("CartesianDiscreteModelsTests.jl") end

@time @testset "DistributedFESpaces" begin include("DistributedFESpacesTests.jl") end

@time @testset "ZeroMeanDistributedFESpacesTests" begin include("ZeroMeanDistributedFESpacesTests.jl") end

@time @testset "DistributedAssemblers" begin include("DistributedAssemblersTests.jl") end

@time @testset "DistributedPoisson" begin include("DistributedPoissonTests.jl") end

@time @testset "DistributedPoissonDG" begin include("DistributedPoissonDGTests.jl") end

@time @testset "DistributedPLaplacian" begin include("DistributedPLaplacianTests.jl") end

@time @testset "DistributedStokes" begin include("DistributedStokesTests.jl") end

end # module
