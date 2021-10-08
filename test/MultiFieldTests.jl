module MultiFieldTests

using Gridap
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays
using Test

function main(parts)

  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(parts,domain,cells)
  Ω = Triangulation(model)

  reffe_u = ReferenceFE(lagrangian,Float64,2)
  reffe_p = ReferenceFE(lagrangian,Float64,1,space=:P)

  V = TestFESpace(model,reffe_u)
  Q = TestFESpace(model,reffe_p)

  VxQ = MultiFieldFESpace([V,Q])

end

end # module
