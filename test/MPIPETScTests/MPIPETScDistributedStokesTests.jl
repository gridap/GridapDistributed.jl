module MPIPETScDistributedStokesTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays
using GridapDistributedPETScWrappers


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
  vector_type = GridapDistributedPETScWrappers.Vec{T}
  matrix_type = GridapDistributedPETScWrappers.Mat{T}

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


  # FE Spaces
  order = 2
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffeₚ = Gridap.ReferenceFEs.LagrangianRefFE(Float64,QUAD,order-1;space=:P)
  V = FESpace(
        vector_type,
        model=model,
        reffe=reffeᵤ,
        conformity=:H1,
        dirichlet_tags=["diri0","diri1"],
      )

  Q = FESpace(
       vector_type,
       model=model,
       reffe=reffeₚ,
       conformity=:L2)
       #constraint=:zeromean)

  Y=MultiFieldFESpace(model,[V,Q])

  U=TrialFESpace(V,[u,u])
  P=TrialFESpace(Q)
  X=MultiFieldFESpace(Y,[U,P])

  if (assembly_strategy == "RowsComputedLocally")
    strategy = RowsComputedLocally(Y; global_dofs=global_dofs)
  elseif (assembly_strategy == "OwnedCellsStrategy")
    strategy = OwnedCellsStrategy(model,Y; global_dofs=global_dofs)
  else
    @assert false "Unknown AssemblyStrategy: $(assembly_strategy)"
  end

  function a(x,y)
    DistributedData(x,y,ddΩ,ddΓ) do part, xl, yl, dΩ, dΓ
        ul,pl=xl
        vl,ql=yl
        ∫( ∇(vl)⊙∇(ul) - (∇⋅vl)*pl + ql*(∇⋅ul) )dΩ
    end
  end

  function l(y)
    DistributedData(y,ddΩ,ddΓ) do part, yl, dΩ, dΓ
      vl,_=yl
      ∫( vl⋅f )dΩ + ∫( vl⋅s )dΓ
    end
  end

   # Assembler
   assem = SparseMatrixAssembler(matrix_type, vector_type, X, Y, strategy)

   function setup_dΩ(part,(model,gids),strategy)
     trian = Triangulation(strategy,model)
     degree = 2*(order+1)
     Measure(trian,degree)
   end
   ddΩ = DistributedData(setup_dΩ,model,strategy)

   function setup_dΓ(part,(model,gids),strategy)
     btrian=BoundaryTriangulation(strategy,model;tags="neumann")
     degree = 2*(order+1)
     Measure(btrian,degree)
   end
   ddΓ = DistributedData(setup_dΓ,model,strategy)


   # FE solution
   op = AffineFEOperator(a,l,X,Y,assem)
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
    dΩ = Measure(owned_trian, 2*order)
    uh,_ = xh
    e = u-uh
    sum(∫(e⋅e)dΩ)
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
