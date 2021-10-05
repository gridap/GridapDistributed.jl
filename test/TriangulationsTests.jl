module TriangulationsTests

using Gridap
using GridapDistributed
using PartitionedArrays
using Test

output = mkpath(joinpath(@__DIR__,"output"))

function main(parts)

  if length(size(parts)) == 2
    domain = (0,4,0,4)
    cells = (4,4)
  elseif length(size(parts)) == 3
    domain = (0,4,0,4,0,4)
    cells = (4,4,4)
  end

  model = CartesianDiscreteModel(parts,domain,cells)

  Ω = Triangulation(with_ghost,model)
  writevtk(Ω,joinpath(output,"Ω"))

  Ω = Triangulation(no_ghost,model)
  writevtk(Ω,joinpath(output,"Ω"))

  Γ = Boundary(with_ghost,model,tags="boundary")
  writevtk(Γ,joinpath(output,"Γ"))

  Γ = Boundary(no_ghost,model,tags="boundary")
  writevtk(Γ,joinpath(output,"Γ"))

end

end # module
