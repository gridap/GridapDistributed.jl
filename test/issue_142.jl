using Gridap
using PartitionedArrays
using GridapDistributed
using Test

function main(distribute,rank_partition)
  DX             = 1000.0
  DY             = 1000.0
  order          = 0
  n_els_x        = 4
  n_els_y        = 4
  dx             = DX/n_els_x
  domain         = (0,DX,0,DY)
  cell_partition = (n_els_x,n_els_y)
  ranks          = distribute(LinearIndices((prod(rank_partition),)))

  model = CartesianDiscreteModel(ranks,rank_partition,domain,cell_partition; isperiodic=(false,false))
  Ω   = Triangulation(model)
  dΩ  = Measure(Ω, 5*(order+1))
  Γ   = SkeletonTriangulation(model)
  dΓ  = Measure(Γ, 5*(order+1))

  Q = FESpace(model, ReferenceFE(lagrangian, Float64, order), conformity=:L2)
  P = TrialFESpace(Q)

  # initial conditions
  ph=FEFunction(P,prand(partition(Q.gids)))

  b(q)   = ∫(jump(ph)*mean(q))dΓ
  m(p,q) = ∫(p*q)dΩ
  op = AffineFEOperator(m, b, P, Q)
  b=assemble_vector(b(get_fe_basis(Q)),P)
  tol=1.0e-12
  @test norm(op.op.vector-b)/norm(b) < tol
end 
