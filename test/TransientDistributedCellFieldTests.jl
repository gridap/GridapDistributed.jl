module TransientDistributedCellFieldTests

using Gridap
using GridapDistributed
using Gridap.ODEs.ODETools: ∂t, ∂tt
using Gridap.ODEs.TransientFETools: TransientCellField
using PartitionedArrays
using Test

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  domain = (0,1,0,1)
  cells = (4,4)
  𝒯 = CartesianDiscreteModel(ranks,parts,domain,cells)
  Ω = Interior(𝒯)
  dΩ = Measure(Ω,2)

  f(t) = t^2
  df(t) = 2t
  ddf(t) = 2

  a(t) = CellField(f(t),Ω)
  da(t) = CellField(df(t),Ω)
  dda(t) = CellField(ddf(t),Ω)
  @test isa(a(0),GridapDistributed.DistributedCellField)
  @test isa(da(0),GridapDistributed.DistributedCellField)
  @test isa(dda(0),GridapDistributed.DistributedCellField)

  b(t) = TransientCellField(a(t),(da(t),dda(t)))
  @test isa(b(0),GridapDistributed.TransientDistributedCellField)
  @test isa(b(0),GridapDistributed.TransientSingleFieldDistributedCellField)

  db(t) = ∂t(b(t))
  @test isa(db(0),GridapDistributed.TransientDistributedCellField)
  @test isa(db(0),GridapDistributed.TransientSingleFieldDistributedCellField)

  ddb(t) = ∂t(db(t))
  @test isa(ddb(0),GridapDistributed.TransientDistributedCellField)
  @test isa(ddb(0),GridapDistributed.TransientSingleFieldDistributedCellField)

  @test (∑(∫(a(0.5))dΩ)) ≈ 0.25
  @test (∑(∫(da(0.5))dΩ)) ≈ 1.0
  @test (∑(∫(dda(0.5))dΩ)) ≈ 2.0
  @test (∑(∫(b(0.5))dΩ)) ≈ 0.25
  @test (∑(∫(db(0.5))dΩ)) ≈ 1.0
  @test (∑(∫(ddb(0.5))dΩ)) ≈ 2.0
  @test (∑(∫(∂tt(b(0.5)))dΩ)) ≈ 2.0
end

end
