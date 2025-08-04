module PeriodicBCsTests

using Gridap
using GridapDistributed
using PartitionedArrays
using Test

function main(distribute,parts)
  ranks  = distribute(LinearIndices((prod(parts),)))
  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,4,0,2π)
  cells = (20,20)
  isperiodic = (false,true)
  model = CartesianDiscreteModel(ranks,parts,domain,cells,isperiodic=isperiodic)

  u((x,y)) = sin(y+π/6)*x
  f(x) = -Δ(u,x)
  k = 2

  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)

  Ω = Interior(model)
  dΩ = Measure(Ω,2*k)

  a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
  l(v) = ∫( v*f )dΩ
  op = AffineFEOperator(a,l,U,V)

  # Test 1: Basic solve(op) functionality
  uh = solve(op)
  eh = u - uh
  error_solve = sqrt(sum( ∫(abs2(eh))dΩ ))
  @test error_solve < 0.00122

  # Test 2: LinearFESolver consistency (fix verification)
  # This tests the specific fix for periodic BCs where LinearFESolver
  # uses compatible vector index structures
  solver = LinearFESolver(BackslashSolver())
  uh_solver = solve(solver, op)
  eh_solver = u - uh_solver
  error_solver = sqrt(sum( ∫(abs2(eh_solver))dΩ ))

  # Test 3: Direct matrix solve for comparison
  A = get_matrix(op)
  b = get_vector(op)
  x_direct = A \ b
  uh_direct = FEFunction(U, x_direct)
  eh_direct = u - uh_direct
  error_direct = sqrt(sum( ∫(abs2(eh_direct))dΩ ))

  # Test 4: Verify all methods give consistent results (within numerical precision)
  # This is the key test for the periodic BC fix - all ratios should be ≈ 1.0
  ratio_solve_direct = error_solve / error_direct
  ratio_solver_direct = error_solver / error_direct

  @test abs(ratio_solve_direct - 1.0) < 1e-10  # solve(op) should match direct solve
  @test abs(ratio_solver_direct - 1.0) < 1e-10  # LinearFESolver should match direct solve
  @test abs(error_solve - error_solver) < 1e-14  # solve(op) and LinearFESolver should be identical

  # Test 5: Error magnitudes should be reasonable
  @test error_solver < 0.00122
  @test error_direct < 0.00122

end

main(DebugArray,(2,2))

end # module
