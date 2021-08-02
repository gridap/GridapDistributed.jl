module MPIPETScDistributedPLaplacianTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays
using LinearAlgebra: norm
using LinearAlgebra
using GridapDistributedPETScWrappers
using NLSolversBase
using NLsolve
using Distances


function NLSolversBase.x_of_nans(x::GridapDistributedPETScWrappers.Vec{Float64}, Tf=eltype(x))
    similar(x)
end

function Base.copyto!(dest::GridapDistributedPETScWrappers.Vec{Float64}, src::GridapDistributedPETScWrappers.Vec{Float64})
   copy!(dest,src)
end

function LinearAlgebra.mul!(Y::GridapDistributedPETScWrappers.Vec{Float64},A::GridapDistributedPETScWrappers.Mat{Float64},B::GridapDistributedPETScWrappers.Vec{Float64})
   GridapDistributedPETScWrappers.C.chk(GridapDistributedPETScWrappers.C.MatMult(A.p,B.p,Y.p))
end

function LinearAlgebra.rmul!(Y::GridapDistributedPETScWrappers.Vec{Float64},A::Number)
  GridapDistributedPETScWrappers.scale!(Y,A)
end

struct PETScVecStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{GridapDistributedPETScWrappers.Vec{Float64}}) = PETScVecStyle()

"`v = find_petscvec(bc)` returns the first GridapDistributedPETScWrappers.Vec among the arguments."
find_petscvec(bc::Base.Broadcast.Broadcasted{PETScVecStyle}) = find_petscvec(bc.args)
find_petscvec(args::Tuple) = find_petscvec(find_petscvec(args[1]), Base.tail(args))
find_petscvec(x) = x
find_petscvec(::Tuple{}) = nothing
find_petscvec(a::GridapDistributedPETScWrappers.Vec{Float64}, rest) = a
find_petscvec(::Any, rest) = find_petscvec(rest)


function Base.similar(bc::Base.Broadcast.Broadcasted{PETScVecStyle}, ::Type{ElType}) where {ElType}
   v=find_petscvec(bc.args)
   similar(v)
end

function Base.copyto!(v::GridapDistributedPETScWrappers.Vec{Float64}, bc::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  @assert bc.f == identity
  @assert length(bc.args)==1
  @assert bc.args[1] isa Number
  fill!(v,Float64(bc.args[1]))
end

function Base.copyto!(v::GridapDistributedPETScWrappers.Vec{Float64}, bc::Base.Broadcast.Broadcasted{PETScVecStyle})
   if bc.f == identity
      @assert length(bc.args)==1
      @assert bc.args[1] isa GridapDistributedPETScWrappers.Vec{Float64}
      copy!(v,bc.args[1])
   elseif bc.f == +
      @assert bc.f == +
      @assert bc.args[1] isa GridapDistributedPETScWrappers.Vec{Float64}
      @assert bc.args[2] isa GridapDistributedPETScWrappers.Vec{Float64}
      GridapDistributedPETScWrappers.C.chk(GridapDistributedPETScWrappers.C.VecWAXPY(v.p, 1.0, bc.args[1].p, bc.args[2].p))
   else
      Gridap.Helpers.@notimplemented
   end
end

function Base.maximum(::typeof(Base.abs), x::GridapDistributedPETScWrappers.Vec{Float64})
  norm(x,Inf)
end

function Base.any(::typeof(Base.isnan),x::GridapDistributedPETScWrappers.Vec{Float64})
   false
end

function NLsolve.check_isfinite(x::GridapDistributedPETScWrappers.Vec{Float64})
   true
end

function (dist::Distances.SqEuclidean)(a::GridapDistributedPETScWrappers.Vec{Float64}, b::GridapDistributedPETScWrappers.Vec{Float64})
  norm(a-b,2)^2
end

function run(comm)
  # Manufactured solution
  u(x) = x[1] + x[2] + 1
  f(x) = - Δ(u)(x)

  p = 3
  flux(∇u) = norm(∇u)^(p-2) * ∇u
  dflux(∇du,∇u) = (p-2)*norm(∇u)^(p-4)*(∇u⋅∇du)*∇u + norm(∇u)^(p-2)*∇du

  # Discretization
  subdomains = (2,2)
  domain = (0,1,0,1)
  cells = (4,4)
  model = CartesianDiscreteModel(comm,subdomains,domain,cells)

  # FE Spaces
  order=3
  degree=2*order
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(model=model,
              reffe=reffe,
              conformity=:H1,
              dirichlet_tags="boundary")
  U = TrialFESpace(V,u)

  trian=Triangulation(model)
  dΩ=Measure(trian,degree)

  function res(u,v)
     ∫(∇(v)⋅(flux∘∇(u)))*dΩ
  end

  function jac(u,du,v)
     ∫( ∇(v)⋅(dflux∘(∇(du),∇(u))) )*dΩ
  end

  # Non linear solver
  ls = PETScLinearSolver(
    Float64;
    ksp_type = "cg",
    ksp_rtol = 1.0e-12,
    ksp_atol = 0.0,
    #ksp_monitor = "",
    pc_type = "jacobi")
  nls = NLSolver(ls; show_trace=true, method=:newton)
  solver = FESolver(nls)

  # FE solution
  op = FEOperator(res,jac,U,V)
  x = zero_free_values(U)
  x .= 1.0
  uh0 = FEFunction(U,x)
  uh, = Gridap.solve!(uh0,solver,op)

  # Error norms and print solution
  trian=Triangulation(OwnedCells,model)
  dΩ=Measure(trian,2*order)
  e = u-uh
  e_l2 = sum(∫(e*e)dΩ)
  tol = 1.0e-9
  @test e_l2 < tol
  if (i_am_master(comm))
    println("$(e_l2) < $(tol)")
  end
end

MPIPETScCommunicator() do comm
  run(comm)
end

end # module
