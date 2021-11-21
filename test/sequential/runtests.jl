module SequentialTests

using Test
using PartitionedArrays
using Gridap

@time @testset "Geometry" begin include("GeometryTests.jl") end

@time @testset "CellData" begin include("CellDataTests.jl") end

@time @testset "FESpaces" begin include("FESpacesTests.jl") end

@time @testset "MultiField" begin include("MultiFieldTests.jl") end

@time @testset "Poisson" begin include("PoissonTests.jl") end

@time @testset "PLaplacian" begin include("PLaplacianTests.jl") end

parts = get_part_ids(sequential,(1,1))
ids   = PRange(parts,6)
vec   = PVector(1.0,ids);
ids2  = PRange(parts,12)
vec2  = PVector(2.0,ids);
vec3vals = map_parts(vec2.owned_values) do ov
  Gridap.Arrays.SubVector(ov,1,6)
end
vec3  = PVector(vec3vals,ids);
@test_broken vec .= vec .+ vec3

end # module
