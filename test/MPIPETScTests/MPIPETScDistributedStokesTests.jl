module MPIPETScDistributedStokesTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays


function Gridap.FESpaces.num_dirichlet_dofs(f::Gridap.MultiField.MultiFieldFESpace)
  result=0
  for space in f.spaces
    result=result+num_dirichlet_dofs(space)
  end
  result
end

function run(comm)

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

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"diri0",[1,2,3,4,6,7,8])
  add_tag_from_tags!(labels,"diri1",[1,2,3,4,6,7,8])
  add_tag_from_tags!(labels,"neumann",[5])

  # FE Spaces
  order = 2
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffeₚ = Gridap.ReferenceFEs.LagrangianRefFE(Float64,QUAD,order-1;space=:P)
  V = FESpace(
        model=model,
        reffe=reffeᵤ,
        conformity=:H1,
        dirichlet_tags=["diri0","diri1"],
      )

  Q = FESpace(
       model=model,
       reffe=reffeₚ,
       conformity=:L2)
       #constraint=:zeromean)

  Y=MultiFieldFESpace(model,[V,Q])

  U=TrialFESpace(V,[u,u])
  P=TrialFESpace(Q)
  X=MultiFieldFESpace(Y,[U,P])

  trian=Triangulation(model)
  degree = 2*(order+1)
  dΩ = Measure(trian,degree)

  btrian=BoundaryTriangulation(model;tags="neumann")
  dΓ = Measure(btrian,degree)

  function a((u,p),(v,q))
    ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
  end

  function l((v,q))
    ∫( v⋅f )dΩ + ∫( v⋅s )dΓ
  end

   # FE solution
   op = AffineFEOperator(a,l,X,Y)
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

  trian=Triangulation(OwnedCells,model)
  dΩ=Measure(trian,2*order)
  uh,_ = xh
  e = u-uh
  e_l2 = sum(∫(e⋅e)dΩ)
  tol = 1.0e-9
  if (i_am_master(comm))
    println("$(e_l2) < $(tol)")
  end
  @test e_l2 < tol
end

MPIPETScCommunicator() do comm
  run(comm)
end

end # module
