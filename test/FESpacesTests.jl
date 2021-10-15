module FESpacesTests

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays
using Test

function main(parts)

  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,1,0,1)
  cells = (4,4)
  model = CartesianDiscreteModel(parts,domain,cells)
  Ω = Triangulation(model)
  Γ = Boundary(model)

  u((x,y)) = x+y
  reffe = ReferenceFE(lagrangian,Float64,1)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)
  @test get_vector_type(V) <: PVector
  @test get_vector_type(U) <: PVector

  free_values = PVector(1.0,V.gids)
  fh = FEFunction(U,free_values)
  zh = zero(U)
  uh = interpolate(u,U)
  eh = u - uh

  dΩ = Measure(Ω,3)
  cont = ∫( abs2(eh) )dΩ
  @test sqrt(sum(cont)) < 1.0e-9

  # Assembly
  dv = get_fe_basis(V)
  du = get_trial_fe_basis(U)
  a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
  l(v) = ∫( 0*dv )dΩ
  assem = SparseMatrixAssembler(U,V)

  data = collect_cell_matrix_and_vector(U,V,a(du,dv),l(dv),zh)
  A,b = assemble_matrix_and_vector(assem,data)
  x = A\b
  r = A*x -b
  uh = FEFunction(U,x)
  eh = u - uh

  writevtk(Ω,joinpath(output,"Ω"), nsubcells=10,
    celldata=["err"=>cont[Ω]],
    cellfields=["uh"=>uh,"zh"=>zh,"eh"=>eh])

  writevtk(Γ,joinpath(output,"Γ"),cellfields=["uh"=>uh])

  @test sqrt(sum(∫( abs2(eh) )dΩ)) < 1.0e-9

  op = AffineFEOperator(a,l,U,V)
  solver = LinearFESolver(BackslashSolver())
  uh = solve(solver,op)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩ)) < 1.0e-9

  data = collect_cell_matrix(U,V,a(du,dv))
  A2 = assemble_matrix(assem,data)

  Ωl = Triangulation(with_ghost,model)
  dΩl = Measure(Ωl,3)
  al(u,v) = ∫( ∇(v)⋅∇(u) )dΩl
  ll(v) = ∫( 0*v )dΩl

  assem = SparseMatrixAssembler(U,V,FullyAssembledRows())
  data = collect_cell_matrix_and_vector(U,V,al(du,dv),ll(dv),zh)
  A,b = assemble_matrix_and_vector(assem,data)
  x = A\b
  r = A*x -b
  uh = FEFunction(U,x)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩ)) < 1.0e-9

  op = AffineFEOperator(al,ll,U,V,FullyAssembledRows())
  uh = solve(solver,op)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩ)) < 1.0e-9

  dv = get_fe_basis(V)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,3)
  l=∫(1*dv)dΩ
  vecdata=collect_cell_vector(V,l)
  assem = SparseMatrixAssembler(U,V,SubAssembledRows())
  b=assemble_vector(assem,vecdata)
  @test sum(b)-1.0 < 1.0e-12

end

end # module
