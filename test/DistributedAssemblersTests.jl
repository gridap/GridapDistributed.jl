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

  strategy = RowsComputedLocally(V; global_dofs=false)
  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)
  function setup_measures(part,(model,gids))
    trian = Triangulation(model)
    degree = 2
    Measure(trian,degree)
  end
  dmeasures = DistributedData(setup_measures,model)

  veccont = DistributedData(V,dmeasures) do part, (VL, gids), dΩ
     vl = get_fe_basis(VL)
     ∫(1*vl)dΩ
  end

  matcont = DistributedData(U,V,dmeasures) do part, (UL,Ugids), (VL,Vgids), dΩ
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

  #TO-DO test_assembler(assem,matdata,vecdata,data)
end


subdomains = (2,2)
SequentialCommunicator(run_simulation,subdomains)

end # module
