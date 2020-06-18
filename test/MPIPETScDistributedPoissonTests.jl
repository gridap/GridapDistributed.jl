module DistributedPoissonTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays
using PETSc

function run(assembly_strategy::AbstractString)
  T = Float64
  fe_space_vector_type = GridapDistributed.MPIPETScDistributedVector{
    T,
    Vector{T},
    Vector{Int},
    Vector{Int},
    Dict{Int,Int32},
  }
  assembler_global_vector_type = PETSc.Vec{T}
  assembler_global_matrix_type = PETSc.Mat{T}
  assembler_local_vector_type = Vector{T}
  assembler_local_matrix_type = SparseMatrixCSR{1,Float64,Int32}

  # Manufactured solution
  u(x) = x[1] + x[2]
  f(x) = -Δ(u)(x)

  # Discretization
  subdomains = (2, 2)
  domain = (0, 1, 0, 1)
  cells = (4, 4)
  comm = MPIPETScCommunicator()
  model = CartesianDiscreteModel(comm, subdomains, domain, cells)

  # FE Spaces
  order = 1
  V = FESpace(
    fe_space_vector_type,
    valuetype = Float64,
    reffe = :Lagrangian,
    order = order,
    model = model,
    conformity = :H1,
    dirichlet_tags = "boundary",
  )

  U = TrialFESpace(V, u)


  if (assembly_strategy == "RowsComputedLocally")
    strategy = RowsComputedLocally(V; global_dofs=false)
  elseif (assembly_strategy == "OwnedCellsStrategy")
    strategy = OwnedCellsStrategy(model, V; global_dofs=false)
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

  # # Assembler
  assem = SparseMatrixAssembler(
    assembler_global_matrix_type,
    assembler_global_vector_type,
    assembler_local_matrix_type,
    assembler_local_vector_type,
    U,
    V,
    strategy,
  )

  # FE solution
  op = AffineFEOperator(assem, terms)
  ls = PETScLinearSolver(
    Float64;
    ksp_type = "cg",
    ksp_rtol = 1.0e-06,
    ksp_atol = 0.0,
    ksp_monitor = "",
    pc_type = "jacobi",
  )
  fels = LinearFESolver(ls)
  uh = solve(fels, op)

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
  if (i_am_master(comm)) println("$(e_l2) < $(tol)\n") end
end

run("RowsComputedLocally")
run("OwnedCellsStrategy")

end # module
