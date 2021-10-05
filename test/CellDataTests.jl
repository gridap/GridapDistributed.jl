module CellDataTests

using Gridap
using GridapDistributed
using PartitionedArrays

function main(parts)
  @assert length(size(parts)) == 2

  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(parts,domain,cells)
  Ω = Triangulation(no_ghost,model)
  Γ = Boundary(no_ghost,model,tags="boundary")

  f = CellField(x->x[1]+x[2],Ω)

  writevtk(Ω,joinpath(output,"Ω"),cellfields=["f"=>f])
  writevtk(Γ,joinpath(output,"Γ"),cellfields=["f"=>f])

end

end # module
