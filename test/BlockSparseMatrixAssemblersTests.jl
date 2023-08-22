module BlockSparseMatrixAssemblersTests

using Test, LinearAlgebra, BlockArrays, SparseArrays

using Gridap
using Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField

using GridapDistributed
using PartitionedArrays

function LinearAlgebra.mul!(y::BlockVector,A::BlockMatrix,x::BlockVector)
  o = one(eltype(A))
  for i in blockaxes(A,2)
    fill!(y[i],0.0)
    for j in blockaxes(A,2)
      mul!(y[i],A[i,j],x[j],o,o)
    end
  end
end

function GridapDistributed.change_ghost(
  x::BlockVector,
  X::GridapDistributed.DistributedMultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}}) where {NB,SB,P}
  block_ranges = MultiField.get_block_ranges(NB,SB,P)
  array = map(block_ranges,blocks(x)) do range, xi
    Xi = (length(range) == 1) ? X.field_fe_space[range[1]] : MultiFieldFESpace(X.field_fe_space[range])
    GridapDistributed.change_ghost(xi,Xi.gids)
  end
  return mortar(array)
end

function is_same_vector(x::BlockVector,y::PVector,Ub,U)
  y_fespace = GridapDistributed.change_ghost(y,U.gids)
  x_fespace = GridapDistributed.change_ghost(x,Ub)

  res = map(1:num_fields(Ub)) do i
    xi = restrict_to_field(Ub,x_fespace,i)
    yi = restrict_to_field(U,y_fespace,i)
    xi ≈ yi
  end
  return all(res)
end

function is_same_matrix(Ab::BlockMatrix,A::PSparseMatrix,Xb,X)
  yb = mortar(map(Aii->pfill(0.0,partition(axes(Aii,1))),diag(blocks(Ab))));
  xb = mortar(map(Aii->pfill(1.0,partition(axes(Aii,2))),diag(blocks(Ab))));
  mul!(yb,Ab,xb)

  y = pfill(0.0,partition(axes(A)[1]))
  x = pfill(1.0,partition(axes(A)[2]))
  mul!(y,A,x)

  return is_same_vector(yb,y,Xb,X)
end

nparts = (2,2)
parts = with_debug() do distribute
  distribute(LinearIndices((prod(nparts),)))
end

sol(x) = sum(x)

model = CartesianDiscreteModel(parts,nparts,(0.0,1.0,0.0,1.0),(8,8))
Ω = Triangulation(model)

reffe = LagrangianRefFE(Float64,QUAD,1)
V = FESpace(Ω, reffe)
U = TrialFESpace(sol,V)

dΩ = Measure(Ω, 4)
biform((u1,u2,u3),(v1,v2,v3)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 + u1⋅v2 + u2⋅v1 + v3⋅u3 + v3⋅u1 + v1⋅u3)*dΩ
liform((v1,v2,v3)) = ∫(v1 + v2 + v3)*dΩ
#biform((u1,u2),(v1,v2)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 + u1⋅v2 - u2⋅v1)*dΩ
#liform((v1,v2)) = ∫(v1 + v2)*dΩ

############################################################################################
# Normal assembly 

Y = MultiFieldFESpace([V,V,V])
X = MultiFieldFESpace([U,U,U])

u = get_trial_fe_basis(X)
v = get_fe_basis(Y)

data = collect_cell_matrix_and_vector(X,Y,biform(u,v),liform(v))
matdata = collect_cell_matrix(X,Y,biform(u,v))
vecdata = collect_cell_vector(Y,liform(v))  

assem = SparseMatrixAssembler(X,Y,FullyAssembledRows())
A1 = assemble_matrix(assem,matdata)
b1 = assemble_vector(assem,vecdata)
A2,b2 = assemble_matrix_and_vector(assem,data);

############################################################################################
# Block MultiFieldStyle

mfs = BlockMultiFieldStyle()
#mfs = BlockMultiFieldStyle(2,(1,2))

Yb  = MultiFieldFESpace([V,V,V];style=mfs)
Xb  = MultiFieldFESpace([U,U,U];style=mfs)

ub = get_trial_fe_basis(Xb)
vb = get_fe_basis(Yb)

bdata = collect_cell_matrix_and_vector(Xb,Yb,biform(ub,vb),liform(vb))
bmatdata = collect_cell_matrix(Xb,Yb,biform(ub,vb))
bvecdata = collect_cell_vector(Yb,liform(vb))

############################################################################################
# Block Assembly

assem_blocks = SparseMatrixAssembler(Xb,Yb,FullyAssembledRows())
A1_blocks = assemble_matrix(assem_blocks,bmatdata);
b1_blocks = assemble_vector(assem_blocks,bvecdata);
is_same_vector(b1_blocks,b1,Yb,Y)
is_same_matrix(A1_blocks,A1,Xb,X)

op = AffineFEOperator(biform,liform,X,Y)
block_op = AffineFEOperator(biform,liform,Xb,Yb)
is_same_vector(get_vector(block_op),get_vector(op),Yb,Y)
is_same_matrix(get_matrix(block_op),get_matrix(op),Xb,X)

end