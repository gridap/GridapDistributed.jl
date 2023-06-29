module TransientMultiFieldDistributedCellFieldTests

using Gridap
using GridapDistributed
using Gridap.ODEs.ODETools: ∂t, ∂tt
using Gridap.ODEs.TransientFETools: TransientCellField
using Gridap.ODEs.TransientFETools: TransientTrialFESpace, TransientMultiFieldFESpace
using PartitionedArrays
using Test

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  domain = (0,1,0,1)
  cells = (4,4)
  𝒯 = CartesianDiscreteModel(ranks,parts,domain,cells)
  Ω = Interior(𝒯)
  dΩ = Measure(Ω,2)

  reffe = ReferenceFE(lagrangian,Float64,1)
  V = FESpace(𝒯, reffe)
  U = TrialFESpace(V)
  Ut = TransientTrialFESpace(V)
  Y = MultiFieldFESpace([V,V])
  X = MultiFieldFESpace([U,U])
  Xt = TransientMultiFieldFESpace([Ut,Ut])

  f(t) = t^2
  df(t) = 2t
  ddf(t) = 2

  a(t) = interpolate([f(t),f(t)],X)
  da(t) = interpolate([df(t),df(t)],X)
  dda(t) = interpolate([ddf(t),ddf(t)],X)
  @test isa(a(0),GridapDistributed.DistributedMultiFieldFEFunction)
  @test isa(da(0),GridapDistributed.DistributedMultiFieldFEFunction)
  @test isa(dda(0),GridapDistributed.DistributedMultiFieldFEFunction)

  b(t) = TransientCellField(a(t),(da(t),dda(t)))
  @test isa(b(0),GridapDistributed.TransientDistributedCellField)
  @test isa(b(0),GridapDistributed.TransientMultiFieldDistributedCellField)

  db(t) = ∂t(b(t))
  @test isa(db(0),GridapDistributed.TransientDistributedCellField)
  @test isa(db(0),GridapDistributed.TransientMultiFieldDistributedCellField)

  ddb(t) = ∂t(db(t))
  @test isa(ddb(0),GridapDistributed.TransientDistributedCellField)
  @test isa(ddb(0),GridapDistributed.TransientMultiFieldDistributedCellField)

  b1(t) = b(t)[1]
  @test isa(b1(0),GridapDistributed.TransientDistributedCellField)
  @test isa(b1(0),GridapDistributed.TransientSingleFieldDistributedCellField)

  db1(t) = ∂t(b1(t))
  @test isa(db1(0),GridapDistributed.TransientDistributedCellField)
  @test isa(db1(0),GridapDistributed.TransientSingleFieldDistributedCellField)

  ddb1(t) = ∂t(db1(t))
  @test isa(ddb1(0),GridapDistributed.TransientDistributedCellField)
  @test isa(ddb1(0),GridapDistributed.TransientSingleFieldDistributedCellField)

  @test (∑(∫(b(0.5)[1])dΩ)) == (∑(∫(b1(0.5))dΩ))
  @test (∑(∫(db(0.5)[1])dΩ)) == (∑(∫(db1(0.5))dΩ))
  @test (∑(∫(ddb(0.5)[1])dΩ)) == (∑(∫(ddb1(0.5))dΩ))
end

end
