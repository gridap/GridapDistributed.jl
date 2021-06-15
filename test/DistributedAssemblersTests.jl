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

  strategy = RowsComputedLocally(V)
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

  A = assemble_matrix(assem,collect_cell_matrix(U,V,matcont))
  b = assemble_vector(assem,collect_cell_vector(V,veccont))
  @test sum(b) ≈ 1
  @test ones(1,size(A,1))*A*ones(size(A,2)) ≈ [1]

  uh = zero(U)

  matdata = collect_cell_matrix(U,V,matcont)
  vecdata = collect_cell_vector(V,veccont)
  data = collect_cell_matrix_and_vector(U,V,matcont,veccont,uh)
  # TO-DO test_assembler(assem,matdata,vecdata,data)
end


subdomains = (2,2)
SequentialCommunicator(run_simulation,subdomains)

end # module
