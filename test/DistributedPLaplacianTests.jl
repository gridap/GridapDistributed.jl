module DistributedPLaplacianTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays
using LinearAlgebra: norm

function run(comm,subdomains,assembly_strategy::AbstractString, global_dofs::Bool)
  # Select matrix and vector types for discrete problem
  # Note that here we use serial vectors and matrices
  # but the assembly is distributed
  T = Float64
  vector_type = Vector{T}
  matrix_type = SparseMatrixCSC{T,Int}

  # Manufactured solution
  u(x) = x[1] + x[2] + 1
  f(x) = - Δ(u)(x)

  p = 3
  flux(∇u) = norm(∇u)^(p-2) * ∇u
  dflux(∇du,∇u) = (p-2)*norm(∇u)^(p-4)*(∇u⋅∇du)*∇u + norm(∇u)^(p-2)*∇du

  # Discretization

  domain = (0,1,0,1)
  cells = (4,4)
  model = CartesianDiscreteModel(comm,subdomains,domain,cells)

  # FE Spaces
  order=3
  degree=2*order
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(vector_type,
              model=model,
              reffe=reffe,
              conformity=:H1,
              dirichlet_tags="boundary")
  U = TrialFESpace(V,u)

  # Choose parallel assembly strategy
  if (assembly_strategy == "RowsComputedLocally")
    strategy = RowsComputedLocally(V;global_dofs=global_dofs)
  elseif (assembly_strategy == "OwnedCellsStrategy"; globals_dofs=global_dofs)
    strategy = OwnedCellsStrategy(model, V; global_dofs=global_dofs)
  else
    @assert false "Unknown AssemblyStrategy: $(assembly_strategy)"
  end

  function setup_dΩ(part,(model,gids),strategy)
    trian = Triangulation(strategy,model)
    Measure(trian,degree)
  end
  ddΩ = DistributedData(setup_dΩ,model,strategy)

  function res(u,v)
    DistributedData(u,v,ddΩ) do part, ul, vl, dΩ
      ∫( ∇(vl)⋅(flux∘∇(ul)) )*dΩ
    end
  end
  function jac(u,du,v)
    DistributedData(u,du,v,ddΩ) do part, ul,dul,vl, dΩ
      ∫( ∇(vl)⋅(dflux∘(∇(dul),∇(ul))) )*dΩ
    end
  end

  # Assembler
  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)

  # Non linear solver
  nls = NLSolver(show_trace=true, method=:newton)
  solver = FESolver(nls)

  # FE solution
  op = FEOperator(res,jac,U,V,assem)
  x = rand(T,num_free_dofs(U))
  uh0 = FEFunction(U,x)
  uh, = solve!(uh0,solver,op)

  # Error norms and print solution
  sums = DistributedData(model, uh) do part, (model, gids), uh
    trian = Triangulation(model)
    owned_trian = remove_ghost_cells(trian, part, gids)
    dΩ = Measure(owned_trian, degree)
    e = u-uh
    sum(∫(e*e)dΩ)
  end
  e_l2 = sum(gather(sums))
  tol = 1.0e-9
  println("$(e_l2) < $(tol)")
  @test e_l2 < tol
end

subdomains = (2,2)
SequentialCommunicator(subdomains) do comm
  run(comm,subdomains,"RowsComputedLocally",true)
  run(comm,subdomains,"OwnedCellsStrategy",true)
end

end # module
