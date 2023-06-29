module FESpacesTests

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays
using Test

u((x,y)) = x+y

function assemble_tests(das,dΩ,dΩass,U,V)
  # Assembly
  dv = get_fe_basis(V)
  du = get_trial_fe_basis(U)
  a(u,v) = ∫( ∇(v)⋅∇(u) )dΩass
  l(v) = ∫( 0*dv )dΩass
  assem = SparseMatrixAssembler(U,V,das)
  zh = zero(U)

  data = collect_cell_matrix_and_vector(U,V,a(du,dv),l(dv),zh)
  A1,b1 = assemble_matrix_and_vector(assem,data)
  x1 = A1\b1
  r1 = A1*x1 -b1
  uh1 = FEFunction(U,x1)
  eh1 = u - uh1
  @test sqrt(sum(∫( abs2(eh1) )dΩ)) < 1.0e-9

  map(A1.matrix_partition, A1.row_partition, A1.col_partition) do mat, rows, cols
     @test size(mat) == (local_length(rows),local_length(cols))
  end

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

function main(distribute,parts)
  main(distribute,parts,SubAssembledRows())
  main(distribute,parts,FullyAssembledRows())
end

function main(distribute,parts,das)
  ranks = distribute(LinearIndices((prod(parts),)))
  
  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,cells)
  Ω = Triangulation(model)
  Γ = Boundary(model)

  u((x,y)) = x+y
  reffe = ReferenceFE(lagrangian,Float64,1)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)
  V2 = FESpace(Ω,reffe)
  @test get_vector_type(V) <: PVector
  @test get_vector_type(U) <: PVector
  @test get_vector_type(V2) <: PVector

  free_values_partition=map(partition(V.gids)) do indices 
    ones(Float64,local_length(indices))
  end 

  free_values = PVector(free_values_partition,partition(V.gids))
  fh = FEFunction(U,free_values)
  zh = zero(U)
  uh = interpolate(u,U)
  eh = u - uh

  uh_dir = interpolate_dirichlet(u,U)
  free_values = zero_free_values(U)
  dirichlet_values = get_dirichlet_dof_values(U)
  uh_dir2 = interpolate_dirichlet!(u,free_values,dirichlet_values,U)

  uh_everywhere = interpolate_everywhere(u,U)
  dirichlet_values0 = zero_dirichlet_values(U)
  uh_everywhere_ = interpolate_everywhere!(u,free_values,dirichlet_values0,U)
  eh2 = u - uh_everywhere
  eh2_ = u - uh_everywhere_

  uh_everywhere2 = interpolate_everywhere(uh_everywhere,U)
  uh_everywhere2_ = interpolate_everywhere!(uh_everywhere,free_values,dirichlet_values,U)
  eh3 = u - uh_everywhere2

  dofs      = get_fe_dof_basis(U)
  cell_vals = dofs(uh)
  gather_free_values!(free_values,U,cell_vals)
  gather_free_and_dirichlet_values!(free_values,dirichlet_values,U,cell_vals)
  uh4 = FEFunction(U,free_values,dirichlet_values)
  eh4 = u - uh4

  dΩ = Measure(Ω,3)
  cont   = ∫( abs2(eh) )dΩ
  cont2  = ∫( abs2(eh2) )dΩ
  cont2_ = ∫( abs2(eh2_) )dΩ
  cont3  = ∫( abs2(eh3) )dΩ
  cont4  = ∫( abs2(eh4) )dΩ
  @test sqrt(sum(cont))   < 1.0e-9
  @test sqrt(sum(cont2))  < 1.0e-9
  @test sqrt(sum(cont2_)) < 1.0e-9
  @test sqrt(sum(cont3))  < 1.0e-9
  @test sqrt(sum(cont4))  < 1.0e-9


  writevtk(Ω,joinpath(output,"Ω"), nsubcells=10,
           celldata=["err"=>cont[Ω]],
           cellfields=["uh"=>uh,"zh"=>zh,"eh"=>eh])

  writevtk(Γ,joinpath(output,"Γ"),cellfields=["uh"=>uh])

  # Assembly
  Ωass  = Triangulation(das,model)
  dΩass = Measure(Ωass,3)
  assemble_tests(das,dΩ,dΩass,U,V)

  u2((x,y)) = 2*(x+y)
  TrialFESpace!(U,u2)
  u2h = interpolate(u2,U)
  e2h = u2 - u2h
  cont  = ∫( abs2(e2h) )dΩ
  @test sqrt(sum(cont)) < 1.0e-9

  U0 = HomogeneousTrialFESpace(U)
  u0h = interpolate(0.0,U0)
  cont  = ∫( abs2(u0h) )dΩ
  @test sqrt(sum(cont)) < 1.0e-14


  # I need to use the square [0,2]² in the sequel so that
  # when integrating over the interior facets, the entries
  # of the vector which is assembled in assemble_tests(...)
  # become one.
  domain = (0,2,0,2)
  cells = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,cells)
  D     = num_cell_dims(model)
  Γ     = Triangulation(ReferenceFE{D-1},model)
  Γass  = Triangulation(das,ReferenceFE{D-1},model)
  dΓ    = Measure(Γ,3)
  dΓass = Measure(Γass,3)
  V = TestFESpace(Γ,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)
  assemble_tests(das,dΓ,dΓass,U,V)

end

end # module
