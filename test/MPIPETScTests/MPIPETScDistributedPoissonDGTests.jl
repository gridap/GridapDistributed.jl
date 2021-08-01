module MPIPETScDistributedPoissonDGTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays
using GridapDistributedPETScWrappers

# Select matrix and vector types for discrete problem
# Note that here we use serial vectors and matrices
# but the assembly is distributed
const T = Float64
const vector_type = GridapDistributedPETScWrappers.Vec{T}
const matrix_type = GridapDistributedPETScWrappers.Mat{T}

# Manufactured solution
function u(x)
  x[1] * (x[1] - 1) * x[2] * (x[2] - 1)
end
f(x) = -Δ(u)(x)
ud(x) = zero(x[1])

# Model
const domain = (0, 1, 0, 1)
const cells  = (4, 4)
const h      = (domain[2] - domain[1]) / cells[1]

# FE Spaces
const order  = 2
const γ = 10
const degree = 2*order

function setup_model(comm)
  # Discretization
  subdomains = (2, 2)
  model = CartesianDiscreteModel(comm, subdomains, domain, cells)
end

function setup_fe_spaces(model)
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(vector_type,
              model=model,
              reffe=reffe,
              conformity=:L2)
  U = TrialFESpace(V)
  U,V
end

function run(comm,model,U,V,strategy)
  trian=Triangulation(strategy,model)
  dΩ=Measure(trian,degree)

  btrian=BoundaryTriangulation(strategy,model)
  dΓ = Measure(btrian,degree)

  strian=SkeletonTriangulation(strategy,model)
  dΛ = Measure(strian,degree)

  n_Γ = get_normal_vector(btrian)
  n_Λ = get_normal_vector(strian)

  function a(u,v)
      ∫( ∇(v)⋅∇(u) )*dΩ +
      ∫( (γ/h)*v*u - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )*dΓ +
      ∫( (γ/h)*jump(v*n_Λ)⋅jump(u*n_Λ) - jump(v*n_Λ)⋅mean(∇(u)) -  mean(∇(v))⋅jump(u*n_Λ) )*dΛ
  end
  function l(v)
      ∫( v*f )*dΩ +
      ∫( (γ/h)*v*ud - (n_Γ⋅∇(v))*ud )*dΓ
  end

  # Assembly
  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)
  op = AffineFEOperator(a,l,U,V,assem)

  # FE solution
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

  trian = Triangulation(OwnedCells,model)
  dΩ = Measure(trian,degree)
  e = u - uh
  l2(u) = ∫( u⊙u )*dΩ
  h1(u) = ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ
  e_l2 = sum(l2(e))
  e_h1 = sum(h1(e))
  tol = 1.0e-9
  @test e_l2 < tol
  @test e_h1 < tol
  if (i_am_master(comm))
    println("$(e_l2) < $(tol)")
    println("$(e_h1) < $(tol)")
  end
end

MPIPETScCommunicator() do comm
  model=setup_model(comm)
  U,V=setup_fe_spaces(model)
  strategy = OwnedAndGhostCellsAssemblyStrategy(V,MapDoFsTypeProcLocal())
  run(comm,model,U,V,strategy)
  strategy = OwnedCellsAssemblyStrategy(V,MapDoFsTypeProcLocal())
  run(comm,model,U,V,strategy)
end

end # module#
