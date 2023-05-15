using Gridap
using Gridap.Geometry
using Gridap.Arrays
using GridapDistributed
using PartitionedArrays

backend = SequentialBackend()
ranks   = (1,2)
parts   = get_part_ids(backend,ranks)

domain    = (0.0,1.0,0.0,1.0)
partition = (3,4)
model     = CartesianDiscreteModel(parts,domain,partition)

order = 0
reffe = ReferenceFE(raviart_thomas,Float64,order)
V = TestFESpace(model,reffe,conformity=:HDiv)
U = TrialFESpace(V)

qdegree = 2*order+1
Ω   = Triangulation(model)
dΩ  = Measure(Ω,qdegree)

Λ   = SkeletonTriangulation(model)
dΛ  = Measure(Λ,qdegree)
n_Λ = get_normal_vector(Λ)

#biform(u,v) = ∫(u⋅v)*dΩ - ∫(jump(v⊗n_Λ)⊙(mean(∇(u))))dΛ

biform(u,v) = ∫((v.minus ⋅ n_Λ.minus) ⋅ (u.plus⋅n_Λ.plus))dΛ
A = assemble_matrix(biform,U,V)
