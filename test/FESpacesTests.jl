module FESpacesTests

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays
using Test

function test_fe_spaces(parts,wghost_or_nghost,das)

  @assert (wghost_or_nghost == with_ghost &&
          isa(das,FullyAssembledRows)) ||
          (wghost_or_nghost == no_ghost &&
          isa(das,SubAssembledRows))

  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,1,0,1)
  cells = (4,4)
  model = CartesianDiscreteModel(parts,domain,cells)
  Ω = Boundary(model)
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

  Ωint  = Triangulation(no_ghost,model)
  dΩint = Measure(Ωint,3)
  cont  = ∫( abs2(eh) )dΩint
  @test sqrt(sum(cont)) < 1.0e-9

  # Assembly
  Ωass  = Triangulation(wghost_or_nghost,model)
  dΩass = Measure(Ωass,3)
  dv = get_fe_basis(V)
  du = get_trial_fe_basis(U)
  a(u,v) = ∫( ∇(v)⋅∇(u) )dΩass
  l(v) = ∫( 0*dv )dΩass
  assem = SparseMatrixAssembler(U,V,das)

  data = collect_cell_matrix_and_vector(U,V,a(du,dv),l(dv),zh)
  A,b = assemble_matrix_and_vector(assem,data)
  x = A\b
  r = A*x -b
  uh = FEFunction(U,x)
  eh = u - uh

  writevtk(Ω,joinpath(output,"Ω"), nsubcells=10,
    celldata=["err"=>cont[Ωint]],
    cellfields=["uh"=>uh,"zh"=>zh,"eh"=>eh])

  writevtk(Γ,joinpath(output,"Γ"),cellfields=["uh"=>uh])

  @test sqrt(sum(∫( abs2(eh) )dΩint)) < 1.0e-9

  op = AffineFEOperator(a,l,U,V,das)
  solver = LinearFESolver(BackslashSolver())
  uh = solve(solver,op)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩint)) < 1.0e-9

  data = collect_cell_matrix(U,V,a(du,dv))
  A2 = assemble_matrix(assem,data)

  al(u,v) = ∫( ∇(v)⋅∇(u) )dΩass
  ll(v) = ∫( 0*v )dΩass

  data = collect_cell_matrix_and_vector(U,V,al(du,dv),ll(dv),zh)
  A,b = assemble_matrix_and_vector(assem,data)
  x = A\b
  r = A*x -b
  uh = FEFunction(U,x)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩint)) < 1.0e-9

  op = AffineFEOperator(al,ll,U,V,das)
  uh = solve(solver,op)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩint)) < 1.0e-9

  dv = get_fe_basis(V)
  l=∫(1*dv)dΩass
  vecdata=collect_cell_vector(V,l)
  assem = SparseMatrixAssembler(U,V,das)
  b=assemble_vector(assem,vecdata)
  @test sum(b)-1.0 < 1.0e-12

end

function main(parts)
  test_fe_spaces(parts,no_ghost,SubAssembledRows())
  test_fe_spaces(parts,with_ghost,FullyAssembledRows())
end

end # module
