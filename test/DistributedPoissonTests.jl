module DistributedPoissonTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays

function run(assembly_strategy::AbstractString, global_dofs::Bool)
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
  order = 1
  V = FESpace(
    vector_type,
    valuetype = Float64,
    reffe = :Lagrangian,
    order = order,
    model = model,
    conformity = :H1,
    dirichlet_tags = "boundary",
  )

  U = TrialFESpace(V, u)

  if (assembly_strategy == "RowsComputedLocally")
    strategy = RowsComputedLocally(V; global_dofs=global_dofs)
  elseif (assembly_strategy == "OwnedCellsStrategy")
    strategy = OwnedCellsStrategy(model,V; global_dofs=global_dofs)
  else
    @assert false "Unknown AssemblyStrategy: $(assembly_strategy)"
  end

  # Terms in the weak form
  terms = DistributedData(model, strategy) do part, (model, gids), strategy
    trian = Triangulation(strategy, model)
    degree = 2 * order
    quad = CellQuadrature(trian, degree)
    a(u, v) = ∇(v) * ∇(u)
    l(v) = v * f
    t1 = AffineFETerm(a, l, trian, quad)
    (t1,)
  end

  # Assembler
  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)

  # FE solution
  op = AffineFEOperator(assem, terms)
  uh = solve(op)

  # Error norms and print solution
  sums = DistributedData(model, uh) do part, (model, gids), uh
    trian = Triangulation(model)
    owned_trian = remove_ghost_cells(trian, part, gids)
    owned_quad = CellQuadrature(owned_trian, 2 * order)
    owned_uh = restrict(uh, owned_trian)
    writevtk(owned_trian, "results_$part", cellfields = ["uh" => owned_uh])
    e = u - owned_uh
    l2(u) = u * u
    sum(integrate(l2(e), owned_trian, owned_quad))
  end
  e_l2 = sum(gather(sums))

  tol = 1.0e-9
  @test e_l2 < tol
end

run("RowsComputedLocally", false)
run("OwnedCellsStrategy", false)
run("RowsComputedLocally", true)
run("OwnedCellsStrategy", true)

end # module
