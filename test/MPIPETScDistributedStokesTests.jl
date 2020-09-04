module DistributedStokesTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays
using PETSc


function Gridap.FESpaces.num_dirichlet_dofs(f::Gridap.MultiField.MultiFieldFESpace)
  result=0
  for space in f.spaces
    result=result+num_dirichlet_dofs(space)
  end
  result
end

function run(comm,assembly_strategy::AbstractString, global_dofs::Bool)
  # Select matrix and vector types for discrete problem
  # Note that here we use serial vectors and matrices
  # but the assembly is distributed
  T = Float64
  vector_type = PETSc.Vec{T}
  matrix_type = PETSc.Mat{T}

  # Manufactured solution
  ux(x)=2*x[1]*x[2]
  uy(x)=-x[2]^2
  u(x)=VectorValue(ux(x),uy(x))
  f(x)=VectorValue(1.0,3.0)
  p(x)=x[1]+x[2]
  sx(x)=-2.0*x[1]
  sy(x)=x[1]+3*x[2]
  s(x)=VectorValue(sx(x),sy(x))

  # Discretization
  domain = (0, 1, 0, 1)
  cells = (4, 4)
  subdomains = (2,2)
  model = CartesianDiscreteModel(comm, subdomains, domain, cells)

  # Define Dirichlet and Neumann boundaries for local models
  do_on_parts(model) do part, (model,gids)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"diri0",[1,2,3,4,6,7,8])
    add_tag_from_tags!(labels,"diri1",[1,2,3,4,6,7,8])
    add_tag_from_tags!(labels,"neumann",[5])
  end

  # Build local and global test FE spaces
  spaces = DistributedData(comm, model) do part, (model,gids)
    labels = get_face_labeling(model)
    V = TestFESpace(
           reffe=:QLagrangian,
           conformity=:H1,
           valuetype=VectorValue{2,Float64},
           model=model,
           labels=labels,
           order=2,
           dirichlet_tags=["diri0","diri1"])
    Q = TestFESpace(
            reffe=:PLagrangian,
            conformity=:L2,
            valuetype=Float64,
            model=model,
            order=1) #,
            #constraint=:zeromean)
    MultiFieldFESpace([V,Q])
  end
  Y=GridapDistributed.DistributedFESpaceFromLocalFESpaces(vector_type,model,spaces)

  # Build local and global trial FE spaces
  trialspaces = DistributedData(comm, Y) do part, (y,gids)
    U=TrialFESpace(y.spaces[1],[u,u])
    P=TrialFESpace(y.spaces[2])
    MultiFieldFESpace([U,P])
  end
  X=GridapDistributed.DistributedFESpaceFromLocalFESpaces(vector_type,trialspaces,Y.gids)

  if (assembly_strategy == "RowsComputedLocally")
    strategy = RowsComputedLocally(Y; global_dofs=global_dofs)
  elseif (assembly_strategy == "OwnedCellsStrategy")
    strategy = OwnedCellsStrategy(model,Y; global_dofs=global_dofs)
  else
    @assert false "Unknown AssemblyStrategy: $(assembly_strategy)"
  end

  # Terms in the weak form
  terms = DistributedData(model, strategy) do part, (model, gids), strategy
    trian = Triangulation(strategy, model)
    btrian=BoundaryTriangulation(strategy,model,"neumann")

    degree=2
    quad=CellQuadrature(trian, degree)
    bquad=CellQuadrature(btrian,degree)

    function a(x,y)
      v,q = y
      y,z = x
      ∇(v)⊙∇(y) - (∇⋅v)*z + q*(∇⋅y)
    end
    function l(y)
      v,q = y
      f⋅v
    end
    function lΓ(y)
      v,q = y
      s⋅v
    end
    t_Ω = AffineFETerm(a,l,trian,quad)
    t_Γ = FESource(lΓ,btrian,bquad)
    (t_Ω,t_Γ)
  end

   # Assembler
   assem = SparseMatrixAssembler(matrix_type, vector_type, X, Y, strategy)

   # FE solution
   op = AffineFEOperator(assem, terms)
   ls = PETScLinearSolver(
    Float64;
    ksp_type = "gmres",
    ksp_rtol = 1.0e-06,
    ksp_atol = 0.0,
    ksp_monitor = "",
    pc_type = "none",
   )
   fels = LinearFESolver(ls)
   xh = solve(fels, op)

  # Error norms and print solution
  sums = DistributedData(model, xh) do part, (model, gids), xh
    trian = Triangulation(model)
    owned_trian = remove_ghost_cells(trian, part, gids)

    owned_quad = CellQuadrature(owned_trian, 2)
    owned_xh = restrict(xh, owned_trian)

    uh, ph = owned_xh

    #writevtk(owned_trian, "results_$part", cellfields = ["uh" => uh])
    e = u - uh
    l2(u) = u ⋅ u
    sum(integrate(l2(e), owned_trian, owned_quad))
  end
  e_l2 = sum(gather(sums))

  tol = 1.0e-9
  if (i_am_master(comm))
    println("$(e_l2) < $(tol)")
  end
  @test e_l2 < tol
end

MPIPETScCommunicator() do comm
  run(comm,"RowsComputedLocally", false)
  run(comm,"OwnedCellsStrategy", false)
end

end # module
