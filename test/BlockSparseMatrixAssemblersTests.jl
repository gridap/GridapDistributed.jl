module BlockSparseMatrixAssemblersTests

using Test, LinearAlgebra, BlockArrays, SparseArrays

using Gridap
using Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField

using GridapDistributed
using PartitionedArrays

nparts = (2,2)
parts = with_debug() do distribute
  distribute(LinearIndices((prod(nparts),)))
end

sol(x) = sum(x)

model = CartesianDiscreteModel(parts,nparts,(0.0,1.0,0.0,1.0),(4,4))
Ω = Triangulation(model)

reffe = LagrangianRefFE(Float64,QUAD,1)
V = FESpace(Ω, reffe)
U = TrialFESpace(sol,V)

dΩ = Measure(Ω, 4)
#biform((u1,u2,u3),(v1,v2,v3)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 + u1⋅v2 - u2⋅v1 - v3⋅u3)*dΩ # + v3⋅u1 - v1⋅u3)*dΩ
#liform((v1,v2,v3)) = ∫(v1 + v2 - v3)*dΩ
biform((u1,u2),(v1,v2)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 + u1⋅v2 - u2⋅v1)*dΩ
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

assem = SparseMatrixAssembler(X,Y,FullyAssembledRows())
A1 = assemble_matrix(assem,matdata)
b1 = assemble_vector(assem,vecdata)
A2,b2 = assemble_matrix_and_vector(assem,data);

assem11 = SparseMatrixAssembler(U,V,FullyAssembledRows())
A11 = assemble_matrix((u1,v1)->∫(∇(u1)⋅∇(v1))*dΩ,assem11,U,V)

############################################################################################
# Block MultiFieldStyle

#mfs = BlockMultiFieldStyle()
mfs = BlockMultiFieldStyle()#2,(1,2))

Yb  = MultiFieldFESpace([V,V];style=mfs)
Xb  = MultiFieldFESpace([U,U];style=mfs)

ub = get_trial_fe_basis(Xb)
vb = get_fe_basis(Yb)

bdata = collect_cell_matrix_and_vector(Xb,Yb,biform(ub,vb),liform(vb))
bmatdata = collect_cell_matrix(Xb,Yb,biform(ub,vb))
bvecdata = collect_cell_vector(Yb,liform(vb))

############################################################################################
# Block Assembly

function LinearAlgebra.mul!(y::BlockVector,A::BlockMatrix,x::BlockVector)
  o = one(eltype(A))
  for i in blockaxes(A,2)
    fill!(y[i],0.0)
    for j in blockaxes(A,2)
      mul!(y[i],A[i,j],x[j],o,o)
    end
  end
end

function is_same_vector(x::BlockVector,y::PVector,Ub,U)
  y_fespace = GridapDistributed.change_ghost(y,U.gids)
  x_fespace = mortar(map((xi,Ui) -> GridapDistributed.change_ghost(xi,Ui.gids),blocks(x),Ub.field_fe_space))

  res = map(1:num_fields(Ub)) do i
    xi = restrict_to_field(Ub,x_fespace,i)
    yi = restrict_to_field(U,y_fespace,i)
    xi ≈ yi
  end
  return all(res)
end

assem_blocks = SparseMatrixAssembler(Xb,Yb,FullyAssembledRows())

A1_blocks = assemble_matrix(assem_blocks,bmatdata);
b1_blocks = assemble_vector(assem_blocks,bvecdata);

y1_blocks = mortar(map(Aii->pfill(0.0,partition(axes(Aii,1))),diag(blocks(A1_blocks))));
x1_blocks = mortar(map(Aii->pfill(1.0,partition(axes(Aii,2))),diag(blocks(A1_blocks))));
mul!(y1_blocks,A1_blocks,x1_blocks)

y1 = pfill(0.0,partition(axes(A1)[1]))
x1 = pfill(1.0,partition(axes(A1)[2]))
mul!(y1,A1,x1)

is_same_vector(y1_blocks,y1,Yb,Y)

############################################################################################

op = AffineFEOperator(biform,liform,X,Y)
block_op = AffineFEOperator(biform,liform,Xb,Yb)


A11 = A1_blocks.blocks[1,1]
A12 = A1_blocks.blocks[1,2]
A22 = A1_blocks.blocks[2,2]

end