module CellDataTests

using Gridap
using GridapDistributed
using PartitionedArrays
using Test

function main(parts)
  @assert length(size(parts)) == 2

  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(parts,domain,cells)
  Ω = Triangulation(model)
  Γ = Boundary(model,tags="boundary")

  f = CellField(x->x[1]+x[2],Ω)
  g = CellField(4.5,Γ)
  v = CellField(x->x,Ω)

  u(x) = sum(x)
  writevtk(Ω,joinpath(output,"Ω"),cellfields=["f"=>f,"u"=>u])
  writevtk(Γ,joinpath(output,"Γ"),cellfields=["f"=>f,"g"=>g,"u"=>u])

  x_Γ = get_cell_points(Γ)
  @test isa(f(x_Γ),AbstractPData)

  h = 4*f
  h = f*g
  h = (v->3*v)∘(f)
  h = ∇(f)
  h = ∇⋅v
  h = ∇×v

  dΩ = Measure(Ω,1)
  dΓ = Measure(Γ,1)

  @test sum( ∫(1)dΩ ) ≈ 16.0
  @test sum( ∫(1)dΓ ) ≈ 16.0
  @test sum( ∫(g)dΓ ) ≈ 4.5*16.0

  x_Γ = get_cell_points(dΓ)
  @test isa(f(x_Γ),AbstractPData)

end

end # module
