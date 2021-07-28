module DistributedPoissonDGTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays

# Select matrix and vector types for discrete problem
# Note that here we use serial vectors and matrices
# but the assembly is distributed
const T = Float64
const vector_type = Vector{T}
const matrix_type = SparseMatrixCSC{T,Int}

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

  function setup_dΩ(part,(model,gids),strategy)
    trian = Triangulation(strategy,model)
    Measure(trian,degree)
  end
  ddΩ = DistributedData(setup_dΩ,model,strategy)

  function setup_dΓ(part,(model,gids),strategy)
    trian = BoundaryTriangulation(strategy,model)
    Measure(trian,degree)
  end
  ddΓ = DistributedData(setup_dΓ,model,strategy)

  function setup_dΛ(part,(model,gids),strategy)
    trian = SkeletonTriangulation(strategy,model)
    Measure(trian,degree)
  end
  ddΛ = DistributedData(setup_dΛ,model,strategy)


  function a(u,v)
    DistributedData(u,v,ddΩ,ddΓ,ddΛ) do part, ul, vl, dΩ, dΓ, dΛ
      n_Γ = get_normal_vector(dΓ.quad.trian)
      n_Λ = get_normal_vector(dΛ.quad.trian)
      ∫( ∇(vl)⋅∇(ul) )*dΩ +
      ∫( (γ/h)*vl*ul - vl*(n_Γ⋅∇(ul)) - (n_Γ⋅∇(vl))*ul )*dΓ +
      ∫( (γ/h)*jump(vl*n_Λ)⋅jump(ul*n_Λ) - jump(vl*n_Λ)⋅mean(∇(ul)) -  mean(∇(vl))⋅jump(ul*n_Λ) )*dΛ
    end
  end
  function l(v)
    DistributedData(v,ddΩ,ddΓ) do part, vl, dΩ, dΓ
      n_Γ = get_normal_vector(dΓ.quad.trian)
      ∫( vl*f )*dΩ +
      ∫( (γ/h)*vl*ud - (n_Γ⋅∇(vl))*ud )*dΓ
    end
  end

  # Assembly
  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)
  op = AffineFEOperator(a,l,U,V,assem)

  # FE solution
  uh = solve(op)

  # Error norms and print solution
  sums = DistributedData(model, uh) do part, (model, gids), uh
    trian = Triangulation(model)
    owned_trian = remove_ghost_cells(trian, part, gids)
    dΩ = Measure(owned_trian, degree)
    e = u - uh
    l2(u) = ∫( u⊙u )*dΩ
    h1(u) = ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ
    e_l2 = sum(l2(e))
    e_h1 = sum(h1(e))
    e_l2, e_h1
  end
  e_l2_h1 = gather(sums)
  e_l2 = sum(map(i -> i[1], e_l2_h1))
  e_h1 = sum(map(i -> i[2], e_l2_h1))
  tol = 1.0e-9
  println("$(e_l2) < $(tol)")
  println("$(e_h1) < $(tol)")
  @test e_l2 < tol
  @test e_h1 < tol
end

subdomains = (2,2)
SequentialCommunicator(subdomains) do comm
  model=setup_model(comm)
  U,V=setup_fe_spaces(model)
  strategy = OwnedAndGhostCellsAssemblyStrategy(V,MapDoFsTypeGlobal())
  run(comm,model,U,V,strategy)
  strategy = OwnedCellsAssemblyStrategy(V,MapDoFsTypeGlobal())
  run(comm,model,U,V,strategy)
end

end # module#
