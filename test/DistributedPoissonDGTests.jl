module DistributedPoissonDGTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays

function run(assembly_strategy::AbstractString)
  # Select matrix and vector types for discrete problem
  # Note that here we use serial vectors and matrices
  # but the assembly is distributed
  T = Float64
  vector_type = Vector{T}
  matrix_type = SparseMatrixCSC{T,Int}

  # Manufactured solution
  u(x) = x[1] * (x[1] - 1) * x[2] * (x[2] - 1)
  f(x) = -Δ(u)(x)
  ud(x) = zero(x[1])

  # Discretization
  subdomains = (2, 2)
  domain = (0, 1, 0, 1)
  cells = (4, 4)
  comm = SequentialCommunicator(subdomains)
  model = CartesianDiscreteModel(comm, subdomains, domain, cells)
  h = (domain[2] - domain[1]) / cells[1]

  # FE Spaces
  order = 2
  V = FESpace(
    vector_type,
    valuetype = Float64,
    reffe = :Lagrangian,
    order = order,
    model = model,
    conformity = :L2,
  )

  U = TrialFESpace(V)

  if (assembly_strategy == "RowsComputedLocally")
    strategy = RowsComputedLocally(V)
  elseif (assembly_strategy == "OwnedCellsStrategy")
    strategy = OwnedCellsStrategy(V)
  else
    @assert false "Unknown AssemblyStrategy: $(assembly_strategy)"
  end

  # Terms in the weak form
  terms = DistributedData(model, strategy) do part, (model, gids), strategy
    γ = 10
    degree = 2 * order

    trian = Triangulation(strategy, model)
    btrian = BoundaryTriangulation(strategy, model)
    strian = SkeletonTriangulation(strategy, model)

    quad = CellQuadrature(trian, degree)
    bquad = CellQuadrature(btrian, degree)
    squad = CellQuadrature(strian, degree)

    bn = get_normal_vector(btrian)
    sn = get_normal_vector(strian)

    a(u, v) = inner(∇(v), ∇(u))
    l(v) = v * f
    t_Ω = AffineFETerm(a, l, trian, quad)

    a_Γd(u, v) = (γ / h) * v * u - v * (bn * ∇(u)) - (bn * ∇(v)) * u
    l_Γd(v) = (γ / h) * v * ud - (bn * ∇(v)) * ud
    t_Γd = AffineFETerm(a_Γd, l_Γd, btrian, bquad)

    a_Γ(u, v) = (γ / h) * jump(v * sn) * jump(u * sn) - jump(v * sn) * mean(∇(u)) -
      mean(∇(v)) * jump(u * sn)
    t_Γ = LinearFETerm(a_Γ, strian, squad)
    (t_Ω, t_Γ, t_Γd)
  end

  # Assembly
  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)
  op = AffineFEOperator(assem, terms)

  # FE solution
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
    h1(u) = u * u + ∇(u) * ∇(u)
    e_l2 = sum(integrate(l2(e), owned_trian, owned_quad))
    e_h1 = sum(integrate(h1(e), owned_trian, owned_quad))
    e_l2, e_h1
  end
  e_l2_h1 = gather(sums)
  e_l2 = sum(map(i -> i[1], e_l2_h1))
  e_h1 = sum(map(i -> i[2], e_l2_h1))
  tol = 1.0e-9
  @test e_l2 < tol
  @test e_h1 < tol
end

run("RowsComputedLocally")
run("OwnedCellsStrategy")

end # module#
