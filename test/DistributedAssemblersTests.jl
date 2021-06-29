module DistributedAssemblersTests

using Gridap
using Gridap.FESpaces
using GridapDistributed
using Test
using SparseArrays


function run_simulation(comm)
  T = Float64
  vector_type = Vector{T}
  matrix_type = SparseMatrixCSC{T,Int}

  domain = (0,1,0,1)
  cells = (4,4)
  model = CartesianDiscreteModel(comm,subdomains,domain,cells)

  reffe = ReferenceFE(lagrangian,Float64,1)
  V = FESpace(vector_type,model=model,reffe=reffe)
  U = TrialFESpace(V)

  strategy = RowsComputedLocally(V; global_dofs=true)
  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)
  function setup_dΩ(part,(model,gids))
    trian = Triangulation(model)
    degree = 2
    Measure(trian,degree)
  end
  ddΩ = DistributedData(setup_dΩ,model)

  veccont = DistributedData(V,ddΩ) do part, (VL, gids), dΩ
     vl = get_fe_basis(VL)
     ∫(1*vl)dΩ
  end

  matcont = DistributedData(U,V,ddΩ) do part, (UL,Ugids), (VL,Vgids), dΩ
    ul = get_trial_fe_basis(UL)
    vl = get_fe_basis(VL)
    ∫(vl*ul)dΩ
  end

  matdata=collect_cell_matrix(U,V,matcont)
  A1 = assemble_matrix(assem,matdata)
  vecdata=collect_cell_vector(V,veccont)
  b1 = assemble_vector(assem,vecdata)
  @test sum(b1) ≈ 1
  @test ones(1,size(A1,1))*A1*ones(size(A1,2)) ≈ [1]

  data = collect_cell_matrix_and_vector(U,V,matcont,veccont)
  A2,b2 = assemble_matrix_and_vector(assem,data)
  @test sum(b2) ≈ 1
  @test ones(1,size(A2,1))*A2*ones(size(A2,2)) ≈ [1]

  @test norm(A1-A2) ≈ 0.0
  @test norm(b1-b2) ≈ 0.0

  A3 = allocate_matrix(assem,matdata)
  @test findnz(A1)[1] == findnz(A3)[1]
  @test findnz(A1)[2] == findnz(A3)[2]
  assemble_matrix!(A3,assem,matdata)
  @test norm(A1-A3) ≈ 0.0

  assemble_matrix_add!(A3,assem,matdata)
  @test norm(2*A1-A3) ≈ 0.0

  b3 = allocate_vector(assem,vecdata)
  assemble_vector!(b3,assem,vecdata)
  @test norm(b3-b1) ≈ 0.0

  assemble_vector_add!(b3,assem,vecdata)
  @test norm(b3-2*b1) ≈ 0.0

  A4, b4 = allocate_matrix_and_vector(assem,data)
  assemble_matrix_and_vector!(A4,b4,assem,data)
  @test norm(A1-A4) ≈ 0.0
  @test norm(b1-b4) ≈ 0.0
  assemble_matrix_and_vector_add!(A4,b4,assem,data)
  @test norm(b4-2*b1) ≈ 0.0
  
end


subdomains = (2,2)
SequentialCommunicator(run_simulation,subdomains)

end # module
