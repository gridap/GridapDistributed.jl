using Test, LinearAlgebra, BlockArrays

using Gridap
using Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField

using GridapDistributed
using PartitionedArrays

parts = get_part_ids(SequentialBackend(),(2,2))

sol(x) = sum(x)

model = CartesianDiscreteModel(parts,(0.0,1.0,0.0,1.0),(6,6))
Ω = Triangulation(model)

reffe = LagrangianRefFE(Float64,QUAD,1)
V = FESpace(Ω, reffe)
U = TrialFESpace(sol,V)

dΩ = Measure(Ω, 2)
biform((u1,u2),(v1,v2)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 + u1⋅v2)*dΩ
liform((v1,v2)) = ∫(v1 + v2)*dΩ

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


touched = MultiField.select_touched_blocks_vecdata(bvecdata,(2,1))


############################################################################################
# Block Assembly

function same_vector(v1::PVector,v2::BlockVector,X)
  v1i = map(i->restrict_to_field(X,v1,i),1:2)
  for i in 1:length(v1i)
    map_parts(v1i[i].owned_values,v2[Block(i)].owned_values) do v1,v2
      @test (norm(v1 - v2) < 1.e-10)
    end
  end
  return true
end

function LinearAlgebra.mul!(y::BlockVector,A::BlockMatrix,x::BlockVector)
  o = one(eltype(A))
  for i in blockaxes(A,1)
    fill!(y[i],0.0)
    for j in blockaxes(A,2)
      mul!(y[i],A[i,j],x[j],o,o)
    end
  end
end

assem_blocks = SparseMatrixAssembler(Xb,Yb)

A1_blocks = assemble_matrix(assem_blocks,bmatdata);
b1_blocks = assemble_vector(assem_blocks,bvecdata);

y1_blocks = mortar(map(Aii->PVector(0.0,Aii.cols),A1_blocks.blocks[1,:]));
x1_blocks = mortar(map(Aii->PVector(1.0,Aii.cols),A1_blocks.blocks[1,:]));

mul!(y1_blocks,A1_blocks,x1_blocks)

y1 = PVector(0.0,A1.cols)
x1 = PVector(1.0,A1.cols)
mul!(y1,A1,x1)

@test same_vector(y1,y1_blocks,X)
@test same_vector(b1,b1_blocks,Y)

############################################################################################

op = AffineFEOperator(biform,liform,X,Y)
block_op = AffineFEOperator(biform,liform,Xb,Yb)

