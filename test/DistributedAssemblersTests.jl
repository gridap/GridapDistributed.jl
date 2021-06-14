module DistributedAssemblersTests

using Gridap
using Gridap.FESpaces
using GridapDistributed
using Test
using SparseArrays

subdomains = (2,2)
SequentialCommunicator(subdomains) do comm
  T = Float64
  vector_type = Vector{T}
  matrix_type = SparseMatrixCSC{T,Int}

  domain = (0,1,0,1)
  cells = (4,4)
  model = CartesianDiscreteModel(comm,subdomains,domain,cells)

  V = FESpace(vector_type,model=model,valuetype=Float64,reffe=:Lagrangian,order=1)

  U = TrialFESpace(V)

  strategy = RowsComputedLocally(V)

  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)

  function setup_terms(part,(model,gids))
    trian = Triangulation(model)
    degree = 2
    quad = CellQuadrature(trian,degree)
    a(u,v) = v*u
    l(v) = 1*v
    t1 = AffineFETerm(a,l,trian,quad)
    (t1,)
  end

  dterms = DistributedData(setup_terms,model)

  vecdata = DistributedData(assem,dterms) do part, assem, terms
    UL = get_trial(assem)
    VL = get_test(assem)
    u0 = zero(UL)
    vl = get_fe_basis(VL)
    collect_cell_vector(u0,vl,terms)
  end

  matdata = DistributedData(assem,dterms) do part, assem, terms
    UL = get_trial(assem)
    VL = get_test(assem)
    ul = get_trial_fe_basis(UL)
    vl = get_fe_basis(VL)
    collect_cell_matrix(ul,vl,terms)
  end

  A = assemble_matrix(assem,matdata)
  b = assemble_vector(assem,vecdata)

  @test sum(b) ≈ 1
  @test ones(1,size(A,1))*A*ones(size(A,2)) ≈ [1]

  v = get_fe_basis(V)
  du = get_trial_fe_basis(U)
  uh = zero(U)

  matdata = collect_cell_matrix(du,v,dterms)
  vecdata = collect_cell_vector(uh,v,dterms)
  data = collect_cell_matrix_and_vector(uh,du,v,dterms)
  test_assembler(assem,matdata,vecdata,data)

  matdata = collect_cell_jacobian(uh,du,v,dterms)
  vecdata = collect_cell_residual(uh,v,dterms)
  data = collect_cell_jacobian_and_residual(uh,du,v,dterms)
  test_assembler(assem,matdata,vecdata,data)
end

end # module
