module FESpacesTests

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays
using Test

function main(parts)
  main(parts,SubAssembledRows())
  main(parts,FullyAssembledRows())
end

function main(parts,das)
  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,4,0,4)
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
  cont  = ∫( abs2(eh) )dΩ
  @test sqrt(sum(cont)) < 1.0e-9

  # Assembly
  Ωass  = Triangulation(das,model)
  dΩass = Measure(Ωass,3)
  dv = get_fe_basis(V)
  du = get_trial_fe_basis(U)
  a(u,v) = ∫( ∇(v)⋅∇(u) )dΩass
  l(v) = ∫( 0*dv )dΩass
  assem = SparseMatrixAssembler(U,V,das)

  data = collect_cell_matrix_and_vector(U,V,a(du,dv),l(dv),zh)
  A1,b1 = assemble_matrix_and_vector(assem,data)
  x1 = A1\b1
  r1 = A1*x1 -b1
  uh1 = FEFunction(U,x1)
  eh1 = u - uh1
  @test sqrt(sum(∫( abs2(eh1) )dΩ)) < 1.0e-9

  writevtk(Ω,joinpath(output,"Ω"), nsubcells=10,
    celldata=["err"=>cont[Ω]],
    cellfields=["uh"=>uh,"zh"=>zh,"eh"=>eh])

  writevtk(Γ,joinpath(output,"Γ"),cellfields=["uh"=>uh])

  A2,b2 = allocate_matrix_and_vector(assem,data)
  assemble_matrix_and_vector!(A2,b2,assem,data)
  x2 = A2\b2
  r2 = A2*x2 -b2
  uh = FEFunction(U,x2)
  eh2 = u - uh
  sqrt(sum(∫( abs2(eh2) )dΩ)) < 1.0e-9

  op = AffineFEOperator(a,l,U,V,das)
  solver = LinearFESolver(BackslashSolver())
  uh = solve(solver,op)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩ)) < 1.0e-9

  data = collect_cell_matrix(U,V,a(du,dv))
  A3 = assemble_matrix(assem,data)
  x3 = A3\op.op.vector
  uh = FEFunction(U,x3)
  eh3 = u - uh
  sqrt(sum(∫( abs2(eh3) )dΩ)) < 1.0e-9

  A4 = allocate_matrix(assem,data)
  assemble_matrix!(A4,assem,data)
  x4 = A4\op.op.vector
  uh = FEFunction(U,x4)
  eh4 = u - uh
  sqrt(sum(∫( abs2(eh4) )dΩ)) < 1.0e-9

  dv = get_fe_basis(V)
  l=∫(1*dv)dΩass
  vecdata=collect_cell_vector(V,l)
  assem = SparseMatrixAssembler(U,V,das)
  b1=assemble_vector(assem,vecdata)
  @test abs(sum(b1)-length(b1)) < 1.0e-12

  b2=allocate_vector(assem,vecdata)
  assemble_vector!(b2,assem,vecdata)
  @test abs(sum(b2)-length(b2)) < 1.0e-12

end

end # module
