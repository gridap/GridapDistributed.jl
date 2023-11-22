
using Gridap
using GridapDistributed
using PartitionedArrays

np = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end


n = 50
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(ranks,np,domain,partition)

labels = get_face_labeling(model);
add_tag_from_tags!(labels,"diri0",[6,])
add_tag_from_tags!(labels,"diri1",[1,2,3,4,5,7,8])

order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["diri0","diri1"])
Q = TestFESpace(model,reffeₚ,conformity=:H1,constraint=:zeromean)

u0(x,t::Real) = t*VectorValue(1,0)
u0(t::Real) = x -> u0(x,t)

u1(x,t::Real) = VectorValue(0.0,0.0)
u1(t::Real) = x -> u1(x,t)

U = TransientTrialFESpace(V,[u0,u1])
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V, Q])
X = TransientMultiFieldFESpace([U, P])

degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

f = VectorValue(0.0,0.0)

m(t,(ut,p),(v,q)) = ∫( ut⋅v )*dΩ
a(t,(u,p),(v,q))  = ∫( ∇(u)⊙∇(v) - (∇⋅v)*p + q*(∇⋅u) )*dΩ
b(t,(v,q))        = ∫( f⋅v )*dΩ

op = TransientAffineFEOperator(m,a,b,X,Y)

dt = 0.1
θ = 0.5
ode_solver = ThetaMethod(LUSolver(),dt,θ)

a₀((u,p),(v,q))  = a(0.0,(u,p),(v,q))
b₀((v,q))        = b(0.0,(v,q))
op₀ = AffineFEOperator(a₀,b₀,X(0.0),Y)
u₀, p₀ = solve(op₀)

x₀ = interpolate_everywhere([u₀,p₀],X(0.0))
t₀ = 0.0
T = 1.0
xₕₜ = solve(ode_solver,op,x₀,t₀,T)

dir = "output/stokes"
!isdir(dir) && mkdir(dir)
for (xₕ,t) in xₕₜ
  (uₕ,pₕ) = xₕ
  file = "output/tmp_stokes_transient_$t.vtu"
  createvtk(Ω,file,cellfields=["uh"=>uₕ,"ph"=>pₕ])
end
