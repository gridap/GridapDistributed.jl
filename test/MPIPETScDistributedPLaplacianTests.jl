module MPIPETScDistributedPLaplacianTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays
using LinearAlgebra: norm
using LinearAlgebra
using PETSc
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

function Gridap.FESpaces._assemble_matrix_and_vector!(A,b,vals_cache,rows_cache,cols_cache,cell_vals,cell_rows,cell_cols,strategy)
  @assert length(cell_cols) == length(cell_rows)
  @assert length(cell_vals) == length(cell_rows)
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    matvals, vecvals = vals
    for (j,gidcol) in enumerate(cols)
      if gidcol > 0 && col_mask(strategy,gidcol)
        _gidcol = col_map(strategy,gidcol)
        for (i,gidrow) in enumerate(rows)
          if gidrow > 0 && row_mask(strategy,gidrow)
            _gidrow = row_map(strategy,gidrow)
            v = matvals[i,j]
            Gridap.Algebra.add_entry!(A,v,_gidrow,_gidcol)
          end
        end
      end
    end
    for (i,gidrow) in enumerate(rows)
      if gidrow > 0 && row_mask(strategy,gidrow)
        _gidrow = row_map(strategy,gidrow)
        bi = vecvals[i]
        Gridap.Algebra.add_entry!(b,bi,_gidrow)
      end
    end
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


function run(comm, assembly_strategy::AbstractString, global_dofs::Bool)
  # Select matrix and vector types for discrete problem
  # Note that here we use serial vectors and matrices
  # but the assembly is distributed
  T = Float64
  vector_type = GridapDistributedPETScWrappers.Vec{T}
  matrix_type = GridapDistributedPETScWrappers.Mat{T}

  # Manufactured solution
  u(x) = x[1] + x[2] + 1
  f(x) = - Δ(u)(x)

  p = 3
  @law flux(∇u) = norm(∇u)^(p-2) * ∇u
  @law dflux(∇du,∇u) = (p-2)*norm(∇u)^(p-4)*(∇u⋅∇du)*∇u + norm(∇u)^(p-2)*∇du

  # Discretization
  subdomains = (2,2)
  domain = (0,1,0,1)
  cells = (4,4)
  model = CartesianDiscreteModel(comm,subdomains,domain,cells)

  # FE Spaces
  order = 3
  V = FESpace(
  vector_type, valuetype=Float64, reffe=:Lagrangian, order=order,
  model=model, conformity=:H1, dirichlet_tags="boundary")

  U = TrialFESpace(V,u)

  # Choose parallel assembly strategy
  if (assembly_strategy == "RowsComputedLocally")
    strategy = RowsComputedLocally(V;global_dofs=global_dofs)
  elseif (assembly_strategy == "OwnedCellsStrategy")
    strategy = OwnedCellsStrategy(model, V; global_dofs=global_dofs)
  else
    @assert false "Unknown AssemblyStrategy: $(assembly_strategy)"
  end

  # Terms in the weak form
  terms = DistributedData(model,strategy) do part, (model,gids), strategy
    trian = Triangulation(strategy,model)
    degree = 2*order
    quad = CellQuadrature(trian,degree)
    res(u,v) = ∇(v)⋅flux(∇(u)) - v*f
    jac(u,du,v) = ∇(v)⋅dflux(∇(du),∇(u))
    t_Ω = FETerm(res,jac,trian,quad)
    (t_Ω,)
  end

  # Assembler
  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)

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
  op = FEOperator(assem,terms)
  x = zero_free_values(U)
  x .= 1.0
  uh0 = FEFunction(U,x)
  uh, = Gridap.solve!(uh0,solver,op)

  # Error norms and print solution
  sums = DistributedData(model,uh) do part, (model,gids), uh
    trian = Triangulation(model)
    owned_trian = remove_ghost_cells(trian,part,gids)
    owned_quad = CellQuadrature(owned_trian,2*order)
    owned_uh = restrict(uh,owned_trian)
    #writevtk(owned_trian,"results_plaplacian_$part",cellfields=["uh"=>owned_uh])
    e = u - owned_uh
    l2(u) = u*u
    sum(integrate(l2(e),owned_trian,owned_quad))
  end
  e_l2 = sum(gather(sums))
  tol = 1.0e-9
  if (i_am_master(comm))
    println("$(e_l2) < $(tol)")
  end
  @test e_l2 < tol
end

MPIPETScCommunicator() do comm
  run(comm,"RowsComputedLocally",false)
  run(comm,"OwnedCellsStrategy",false)
end

end # module
