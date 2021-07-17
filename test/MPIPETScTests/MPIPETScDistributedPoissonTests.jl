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
  order=1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(vector_type,
              model=model,
              reffe=reffe,
              conformity=:H1,
              dirichlet_tags="boundary")
  U = TrialFESpace(V, u)


  if (assembly_strategy == "RowsComputedLocally")
    strategy = RowsComputedLocally(V; global_dofs=false)
  elseif (assembly_strategy == "OwnedCellsStrategy")
    strategy = OwnedCellsStrategy(model, V; global_dofs=false)
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
    dΩ = Measure(owned_trian, 2*order)
    e = u-uh
    sum(∫(e*e)dΩ)
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
