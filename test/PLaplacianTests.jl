module PLaplacianTests
using SparseMatricesCSR
using Gridap
using Gridap.Algebra
using GridapDistributed
using PartitionedArrays
using Test
using SparseArrays

function main(distribute,parts)
  main(distribute,parts,LocallyAssembled(),SparseMatrixCSR{0,Float64,Int},false)
  main(distribute,parts,Assembled(),SparseMatrixCSC{Float64,Int},true)
end

function main(distribute,parts,strategy,local_matrix_type,autodiff)
  ranks  = distribute(LinearIndices((prod(parts),)))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,cells)

  k = 1
  u((x,y)) = (x+y)^k
  σ(∇u) = (1.0+∇u⋅∇u)*∇u
  dσ(∇du,∇u) = (2*∇u⋅∇du)*∇u + (1.0+∇u⋅∇u)*∇du
  f(x) = -divergence(y->σ(∇(u,y)),x)

  Ω = Triangulation(strategy,model)
  dΩ = Measure(Ω,2*k)
  r(u,v) = ∫( ∇(v)⋅(σ∘∇(u)) - v*f )dΩ
  j(u,du,v) = ∫( ∇(v)⋅(dσ∘(∇(du),∇(u))) )dΩ

  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)

  assem = SparseMatrixAssembler(local_matrix_type,Vector{Float64},U,V,strategy)
  if !autodiff
    op = FEOperator(r,j,U,V,assem)
  else
    op = FEOperator(r,U,V,assem)
  end

  uh = zero(U)
  b,A = residual_and_jacobian(op,uh)
  _A = copy(A)
  _b = copy(b)
  residual_and_jacobian!(_b,_A,op,uh)
  @test (norm(_b-b)+1) ≈ 1
  x = similar(b,Float64,axes(A,2))
  fill!(x,1)
  @test (norm(A*x-_A*x)+1) ≈ 1

  # This leads to a dead lock since the printing of the trace seems to lead to collective operations
  # show_trace = i_am_main(parts)
  show_trace = true # The output in MPI will be ugly, but fixing this would require to edit NLSolvers package.
  nls = NLSolver(show_trace=show_trace, method=:newton)
  solver = FESolver(nls)
  uh = solve(solver,op)

  Ωo = Triangulation(model)
  dΩo = Measure(Ωo,2*k)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩo)) < 1.0e-9
end

end # module
