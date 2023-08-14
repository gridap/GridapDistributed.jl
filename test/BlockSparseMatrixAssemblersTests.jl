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

model = CartesianDiscreteModel(parts,nparts,(0.0,1.0,0.0,1.0),(12,12))
Ω = Triangulation(model)

reffe = LagrangianRefFE(Float64,QUAD,1)
V = FESpace(Ω, reffe)
U = TrialFESpace(sol,V)

dΩ = Measure(Ω, 4)
biform((u1,u2,u3),(v1,v2,v3)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 + u1⋅v2 - u2⋅v1 - v3⋅u3)*dΩ # + v3⋅u1 - v1⋅u3)*dΩ
liform((v1,v2,v3)) = ∫(v1 + v2 - v3)*dΩ

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

assem11 = SparseMatrixAssembler(U,V,FullyAssembledRows())
A11 = assemble_matrix((u1,v1)->∫(∇(u1)⋅∇(v1))*dΩ,assem11,U,V)

############################################################################################
# Block MultiFieldStyle

#mfs = BlockMultiFieldStyle()
mfs = BlockMultiFieldStyle(2,(1,2))

Yb  = MultiFieldFESpace([V,V,V];style=mfs)
Xb  = MultiFieldFESpace([U,U,U];style=mfs)

ub = get_trial_fe_basis(Xb)
vb = get_fe_basis(Yb)

bdata = collect_cell_matrix_and_vector(Xb,Yb,biform(ub,vb),liform(vb))
bmatdata = collect_cell_matrix(Xb,Yb,biform(ub,vb))
bvecdata = collect_cell_vector(Yb,liform(vb))

############################################################################################
# Block Assembly

function same_solution(x1::PVector,x2::BlockVector,X,Xi,dΩ)
  u1 = [FEFunction(X,x1)...]
  u2 = map(i->FEFunction(Xi[i],x2[Block(i)]),1:blocklength(x2))

  err = map(u1,u2) do u1,u2
    eh = u1-u2
    return sum(∫(eh⋅eh)dΩ)
  end
  return err
end

function LinearAlgebra.mul!(y::BlockVector,A::BlockMatrix,x::BlockVector)
  o = one(eltype(A))
  for i in blockaxes(A,2)
    fill!(y[i],0.0)
    for j in blockaxes(A,2)
      mul!(y[i],A[i,j],x[j],o,o)
    end
  end
end

function test_axes(c::BlockVector,a::BlockMatrix,b::BlockVector)
  res = Matrix(undef,blocksize(a)...)
  for i in blockaxes(a,1)
    for j in blockaxes(a,2)
      res[i.n[1],j.n[1]] = Tuple([oids_are_equal(c[i].rows,a[i,j].rows),
      oids_are_equal(a[i,j].cols,b[j].rows),
      hids_are_equal(a[i,j].cols,b[j].rows)])
    end
  end
  return res
end

function get_block_fespace(spaces,range)
  (length(range) == 1) ? spaces[range[1]] : MultiFieldFESpace(spaces[range])
end

block_ranges = Gridap.MultiField.get_block_ranges(2,(1,2),(1,2,3))
block_trials = map(range -> get_block_fespace(X.field_fe_space,range),block_ranges)

#! TODO: Does not work if there are empty blocks due to PRange checks when multiplying. 
#! Maybe we should change to MatrixBlocks?  

assem_blocks = SparseMatrixAssembler(Xb,Yb,FullyAssembledRows())

local_views(assem_blocks)

ab = assem_blocks.block_assemblers
map(local_views,ab)

A1_blocks = assemble_matrix(assem_blocks,bmatdata);
b1_blocks = assemble_vector(assem_blocks,bvecdata);

y1_blocks = mortar(map(Aii->PVector(0.0,Aii.rows),diag(A1_blocks.blocks)));
x1_blocks = mortar(map(Aii->PVector(1.0,Aii.cols),diag(A1_blocks.blocks)));
test_axes(y1_blocks,A1_blocks,x1_blocks)

mul!(y1_blocks,A1_blocks,x1_blocks)

y1 = PVector(0.0,A1.rows)
x1 = PVector(1.0,A1.cols)
mul!(y1,A1,x1)

@test all(same_solution(y1,y1_blocks,X,block_trials,dΩ) .< 1e-5)

############################################################################################

op = AffineFEOperator(biform,liform,X,Y)
block_op = AffineFEOperator(biform,liform,Xb,Yb)


A11 = A1_blocks.blocks[1,1]
A12 = A1_blocks.blocks[1,2]
A22 = A1_blocks.blocks[2,2]

end