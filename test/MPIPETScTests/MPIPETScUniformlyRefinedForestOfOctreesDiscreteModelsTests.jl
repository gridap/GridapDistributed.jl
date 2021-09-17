module MPIPETScUniformlyRefinedForestOfOctreesDiscreteModelsTests
  using Gridap
  using GridapDistributed
  using Test
  using ArgParse



  function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--subdomains", "-s"
        help = "Tuple with the # of coarse subdomains per Cartesian direction"
        arg_type = Int64
        default=[1,1]
        nargs='+'
        "--num-uniform-refinements", "-r"
        help = "# of uniform refinements"
        arg_type = Int64
        default=1
    end
    return parse_args(s)
  end


  function run(comm,subdomains,num_uniform_refinements)
    # Manufactured solution
    u(x) = x[1] + x[2]
    f(x) = -Δ(u)(x)

    if length(subdomains)==2
      domain=(0,1,0,1)
    else
      @assert length(subdomains)==3
      domain=(0,1,0,1,0,1)
    end

    coarse_discrete_model=CartesianDiscreteModel(domain,subdomains)
    model=UniformlyRefinedForestOfOctreesDiscreteModel(comm,
                                                       coarse_discrete_model,
                                                       num_uniform_refinements)

    # FE Spaces
    order=1
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = FESpace(model=model,
              reffe=reffe,
              conformity=:H1,
              dirichlet_tags="boundary")
    U = TrialFESpace(V, u)

    trian=Triangulation(model)
    dΩ=Measure(trian,2*(order+1))

    function a(u,v)
      ∫(∇(v)⋅∇(u))dΩ
    end
    function l(v)
      ∫(v*f)dΩ
    end

    # FE solution
    op = AffineFEOperator(a,l,U,V)
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
    trian=Triangulation(OwnedCells,model)
    dΩ=Measure(trian,2*order)
    e = u-uh
    e_l2 = sum(∫(e*e)dΩ)
    tol = 1.0e-9
    @test e_l2 < tol
    if (i_am_master(comm)) println("$(e_l2) < $(tol)\n") end
  end

  parsed_args = parse_commandline()
  subdomains = Tuple(parsed_args["subdomains"])
  num_uniform_refinements = Tuple(parsed_args["num-uniform-refinements"])

  MPIPETScCommunicator() do comm
    run(comm,subdomains,num_uniform_refinements[1])
  end

end # module
