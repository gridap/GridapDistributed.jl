function Gridap.FESpaces.solve!(u::DistributedFEFunction{<:MPIPETScDistributedVector},solver::LinearFESolver,feop::AffineFEOperator,cache::Nothing)
  @assert is_a_fe_function(u)
  # TO-DO: The fact that I have to ask in execution time for the type of solver.ls
  #        reveals a rotten software design. How could we avoid this? Should LinearFESolver
  #        be paremeterized by the type of LinearSolver?
  @assert typeof(solver.ls) <: PETScLinearSolver
  x = get_free_values(u)
  # TO-DO: The fact that I have to ask in execution time for the type of x
  #        reveals a rotten software design. How could we avoid this?
  @assert typeof(x) <: MPIPETScDistributedVector
  xvec = x.vecghost
  op = get_algebraic_operator(feop)
  cache = Gridap.FESpaces.solve!(xvec,solver.ls,op)
  trial = get_trial(feop)
  u_new = FEFunction(trial,x)
  (u_new, cache)
end
