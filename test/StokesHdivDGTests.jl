module StokesHdivDGTests

using Gridap
using GridapDistributed
using PartitionedArrays
using GridapDistributed
using Gridap.TensorValues
using Test


function l2_error(Ω,u1,u2)
  dΩ = Measure(Ω,10)
  e = u1-u2
  return sum(∫(e⋅e)dΩ)
end

struct SolutionBundle
  u :: Function
  p :: Function
  f :: Function
  σ :: Function
  constants :: Dict
end

function stokes_solution_2D(μ::Real)
  u(x) = VectorValue(x[1],x[2]) 
  p(x) = 2.0*x[1]-1.0

  f(x) = -μ⋅Δ(u)(x) + ∇(p)(x)
  σ(x) = μ⋅∇(u)(x) - p(x)⋅TensorValue(1.0,0.0,0.0,1.0)

  constants = Dict{Symbol,Real}(:μ => μ)
  return SolutionBundle(u,p,f,σ,constants)
end

function main(distribute,parts)
    ranks = distribute(LinearIndices((prod(parts),)))
    
    μ = 1.0
    sol = stokes_solution_2D(μ)
    u_ref = sol.u
    f_ref = sol.f
    σ_ref = sol.σ

    D = 2
    n = 4
    domain    = Tuple(repeat([0,1],D))
    partition = (n,n)
    model     = CartesianDiscreteModel(ranks,parts,domain,partition)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"dirichlet",[5,6,7])
    add_tag_from_tags!(labels,"neumann",[8,])

    ############################################################################################
    order = 1
    reffeᵤ = ReferenceFE(raviart_thomas,Float64,order)
    V = TestFESpace(model,reffeᵤ,conformity=:HDiv,dirichlet_tags="dirichlet")
    U = TrialFESpace(V,u_ref)

    reffeₚ = ReferenceFE(lagrangian,Float64,order;space=:P)
    Q = TestFESpace(model,reffeₚ,conformity=:L2)
    P = TrialFESpace(Q)

    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])

    qdegree = 2*order+1
    Ω   = Triangulation(model)
    dΩ  = Measure(Ω,qdegree)

    Γ   = BoundaryTriangulation(model)
    dΓ  = Measure(Γ,qdegree)
    n_Γ = get_normal_vector(Γ)

    Γ_D  = BoundaryTriangulation(model;tags=["dirichlet"])
    dΓ_D = Measure(Γ_D,qdegree)
    n_Γ_D = get_normal_vector(Γ_D)

    Γ_N  = BoundaryTriangulation(model;tags="neumann")
    dΓ_N = Measure(Γ_N,qdegree)
    n_Γ_N = get_normal_vector(Γ_N)

    Λ   = SkeletonTriangulation(model)
    dΛ  = Measure(Λ,qdegree)
    n_Λ = get_normal_vector(Λ)

    h_e    = CellField(map(get_array,local_views(∫(1)dΩ)),Ω)
    h_e_Λ  = CellField(map(get_array,local_views(∫(1)dΛ)),Λ)
    h_e_Γ_D = CellField(map(get_array,local_views(∫(1)dΓ_D)),Γ_D)

    β_U = 50.0
    Δ_dg(u,v) = ∫(∇(v)⊙∇(u))dΩ - 
                ∫(jump(v⊗n_Λ)⊙(mean(∇(u))))dΛ -
                ∫(mean(∇(v))⊙jump(u⊗n_Λ))dΛ - 
                ∫(v⋅(∇(u)⋅n_Γ_D))dΓ_D - 
                ∫((∇(v)⋅n_Γ_D)⋅u)dΓ_D
    rhs((v,q)) = ∫((f_ref⋅v))*dΩ - ∫((∇(v)⋅n_Γ_D)⋅u_ref)dΓ_D + ∫((n_Γ_N⋅σ_ref)⋅v)*dΓ_N

    penalty(u,v) = ∫(jump(v⊗n_Λ)⊙((β_U/h_e_Λ*jump(u⊗n_Λ))))dΛ + ∫(v⋅(β_U/h_e_Γ_D*u))dΓ_D
    penalty_rhs((v,q)) = ∫(v⋅(β_U/h_e_Γ_D*u_ref))dΓ_D

    a((u,p),(v,q)) = Δ_dg(u,v) + ∫(-(∇⋅v)*p - q*(∇⋅u))dΩ  + penalty(u,v)  
    l((v,q)) = rhs((v,q)) - ∫(q*(∇⋅u_ref))dΩ + penalty_rhs((v,q)) 

    op = AffineFEOperator(a,l,X,Y)
    xh = solve(op)

    uh, ph = xh
    err_u = l2_error(Ω,uh,sol.u) 
    err_p = l2_error(Ω,ph,sol.p)
    tol = 1.0e-12
    @test err_u < tol
    @test err_p < tol
end

end # module
