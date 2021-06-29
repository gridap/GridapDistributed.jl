module DistributedPoissonTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays

function run(comm,subdomains,assembly_strategy::AbstractString, global_dofs::Bool)
  # Select matrix and vector types for discrete problem
  # Note that here we use serial vectors and matrices
  # but the assembly is distributed
  T = Float64
  vector_type = Vector{T}
  matrix_type = SparseMatrixCSC{T,Int}

  # Manufactured solution
  u(x) = x[1] + x[2]
  f(x) = -Δ(u)(x)

  # Discretization
  subdomains = (2, 2)
  domain = (0, 1, 0, 1)
  cells = (4, 4)
  comm = SequentialCommunicator(subdomains)
  model = CartesianDiscreteModel(comm, subdomains, domain, cells)

  # FE Spaces
  order=1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(vector_type,
              model=model,
              reffe=reffe,
              conformity=:H1,
              dirichlet_tags="boundary")
  U = TrialFESpace(V, u)

  if (assembly_strategy == "RowsComputedLocally")
    strategy = RowsComputedLocally(V; global_dofs=global_dofs)
  elseif (assembly_strategy == "OwnedCellsStrategy")
    strategy = OwnedCellsStrategy(model,V; global_dofs=global_dofs)
  else
    @assert false "Unknown AssemblyStrategy: $(assembly_strategy)"
  end

  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)
  function setup_dΩ(part,(model,gids),strategy)
    trian = Triangulation(strategy,model)
    degree = 2*(order+1)
    Measure(trian,degree)
  end
  ddΩ = DistributedData(setup_dΩ,model,strategy)

  function a(u,v)
    DistributedData(u,v,ddΩ) do part, ul, vl, dΩ
      ∫(∇(vl)⋅∇(ul))dΩ
    end
  end
  function l(v)
    DistributedData(v,ddΩ) do part, vl, dΩ
      ∫(vl*f)dΩ
    end
  end

  # FE solution
  op = AffineFEOperator(a,l,U,V,assem)
  uh = solve(op)

  sums = DistributedData(model, uh) do part, (model, gids), uh
    trian = Triangulation(model)
    owned_trian = remove_ghost_cells(trian, part, gids)
    dΩ = Measure(owned_trian, 2*order)
    e = u-uh
    sum(∫(e*e)dΩ)
  end
  e_l2 = sum(gather(sums))
  tol = 1.0e-9
  println("$(e_l2) < $(tol)")
  @test e_l2 < tol
end

subdomains = (2,2)
SequentialCommunicator(subdomains) do comm
  run(comm,subdomains,"RowsComputedLocally", false)
  run(comm,subdomains,"OwnedCellsStrategy", false)
  #run(comm,subdomains,"RowsComputedLocally", true)
  #run(comm,subdomains,"OwnedCellsStrategy", true)
end

end # module
