using Test, LinearAlgebra

using Gridap
using Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField

using GridapDistributed
using PartitionedArrays

parts = get_part_ids(SequentialBackend(),(2,2))

sol(x) = sum(x)

model = CartesianDiscreteModel(parts,(0.0,1.0,0.0,1.0),(10,10))
Ω = Triangulation(model)

reffe = LagrangianRefFE(Float64,QUAD,1)
V = FESpace(Ω, reffe; dirichlet_tags="boundary")
U = TrialFESpace(sol,V)

dΩ = Measure(Ω, 2)
biform((u1,u2),(v1,v2)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 - u1⋅v2)*dΩ
liform((v1,v2)) = ∫(v1 - v2)*dΩ

############################################################################################
# Normal assembly 

Y = MultiFieldFESpace([V,V])
X = MultiFieldFESpace([U,U])

u = get_trial_fe_basis(X)
v = get_fe_basis(Y)

data = collect_cell_matrix_and_vector(X,Y,biform(u,v),liform(v))
matdata = collect_cell_matrix(X,Y,biform(u,v))
vecdata = collect_cell_vector(Y,liform(v))  

assem = SparseMatrixAssembler(X,Y)
A1 = assemble_matrix(assem,matdata)
b1 = assemble_vector(assem,vecdata)
A2,b2 = assemble_matrix_and_vector(assem,data);

############################################################################################
# Block MultiFieldStyle

mfs = BlockMultiFieldStyle()
Yb = MultiFieldFESpace([V,V];style=mfs)
Xb = MultiFieldFESpace([U,U];style=mfs)

ub = get_trial_fe_basis(Xb)
vb = get_fe_basis(Yb)

bdata = collect_cell_matrix_and_vector(Xb,Yb,biform(ub,vb),liform(vb))
bmatdata = collect_cell_matrix(Xb,Yb,biform(ub,vb))
bvecdata = collect_cell_vector(Yb,liform(vb))

assem_blocks = SparseMatrixAssembler(Xb,Yb)
