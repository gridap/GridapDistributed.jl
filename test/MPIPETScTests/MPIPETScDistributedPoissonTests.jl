module MPIPETScDistributedPoissonTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays
using GridapDistributedPETScWrappers

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--subdomains", "-s"
        help = "Tuple with the # of subdomains per Cartesian direction"
        arg_type = Int64
        default=[1,1]
        nargs='+'
        "--partition", "-p"
        help = "Tuple with the # of cells per Cartesian direction"
        arg_type = Int64
        default=[4,4]
        nargs='+'
    end
    return parse_args(s)
end


function run(comm,
             assembly_strategy::AbstractString,
             subdomains=(2, 2),
             cells=(4, 4),
             domain = (0, 1, 0, 1))

  T = Float64
  vector_type = GridapDistributedPETScWrappers.Vec{T}
  matrix_type = GridapDistributedPETScWrappers.Mat{T}

  # Manufactured solution
  u(x) = x[1] + x[2]
  f(x) = -Δ(u)(x)

  # Discretization
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
    a(u, v) = ∇(v)⋅∇(u)
    l(v) = v * f
    t1 = AffineFETerm(a, l, trian, quad)
    (t1,)
  end

  # # Assembler
  assem = SparseMatrixAssembler(
    matrix_type,
    vector_type,
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


parsed_args = parse_commandline()
subdomains = Tuple(parsed_args["subdomains"])
partition = Tuple(parsed_args["partition"])

MPIPETScCommunicator() do comm
  run(comm, "RowsComputedLocally", subdomains,partition)
  run(comm, "OwnedCellsStrategy",subdomains,partition)
end

end # module
