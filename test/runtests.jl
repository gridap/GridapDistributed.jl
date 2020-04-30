module GridapDistributedTests

using Test

@time @testset "DistributedData" begin include("DistributedDataTests.jl") end

@time @testset "DistributedIndexSets" begin include("DistributedIndexSetsTests.jl") end

@time @testset "DistributedVectors" begin include("DistributedVectorsTests.jl") end

@time @testset "CartesianDiscreteModels" begin include("CartesianDiscreteModelsTests.jl") end

#@time @testset "DistributedTriangulations" begin include("DistributedTriangulationsTests.jl") end
#
#@time @testset "DistributedCellQuadratures" begin include("DistributedCellQuadraturesTests.jl") end
#
#@time @testset "DistributedCellFields" begin include("DistributedCellFieldsTests.jl") end

@time @testset "DistributedFESpaces" begin include("DistributedFESpacesTests.jl") end

#@time @testset "GloballyAddressableArrays" begin include("GloballyAddressableArraysTests.jl") end
#
#@time @testset "SparseMatrixAssemblers" begin include("SparseMatrixAssemblersTests.jl") end
#
#@time @testset "DistributedAssemblers" begin include("DistributedAssemblersTests.jl") end

end # module
