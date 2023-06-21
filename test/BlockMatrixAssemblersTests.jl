using Test, LinearAlgebra, BlockArrays

using Gridap
using Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField

using GridapDistributed
using PartitionedArrays

parts = get_part_ids(SequentialBackend(),(2,2))

sol(x) = sum(x)

model = CartesianDiscreteModel(parts,(0.0,1.0,0.0,1.0),(12,12))
Ω = Triangulation(model)

reffe = LagrangianRefFE(Float64,QUAD,1)
V = FESpace(Ω, reffe)
U = TrialFESpace(sol,V)

dΩ = Measure(Ω, 4)
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

assem = SparseMatrixAssembler(X,Y,FullyAssembledRows())
A1 = assemble_matrix(assem,matdata)
b1 = assemble_vector(assem,vecdata)
A2,b2 = assemble_matrix_and_vector(assem,data);

assem11 = SparseMatrixAssembler(U,V,FullyAssembledRows())
A11 = assemble_matrix((u1,v1)->∫(∇(u1)⋅∇(v1))*dΩ,assem11,U,V)

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

############################################################################################
# Block Assembly

function same_solution(x1::PVector,x2::BlockVector,X,dΩ)
  u1 = [FEFunction(X,x1)...]
  u2 = map(i->FEFunction(X[i],x2[Block(i)]),1:blocklength(x2))

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
  tests = []
  for i in blockaxes(a,1)
    for j in blockaxes(a,2)
      push!(tests,
      (oids_are_equal(c[i].rows,a[i,j].rows),
      oids_are_equal(a[i,j].cols,b[j].rows),
      hids_are_equal(a[i,j].cols,b[j].rows)))
    end
  end
  return tests
end

assem_blocks = SparseMatrixAssembler(Xb,Yb,FullyAssembledRows())

A1_blocks = assemble_matrix(assem_blocks,bmatdata);
b1_blocks = assemble_vector(assem_blocks,bvecdata);

y1_blocks = mortar(map(Aii->PVector(0.0,Aii.rows),A1_blocks.blocks[:,1]));
x1_blocks = mortar(map(Aii->PVector(1.0,Aii.cols),A1_blocks.blocks[1,:]));
test_axes(y1_blocks,A1_blocks,x1_blocks)

mul!(y1_blocks,A1_blocks,x1_blocks)

y1 = PVector(0.0,A1.rows)
x1 = PVector(1.0,A1.cols)
mul!(y1,A1,x1)

@test all(same_solution(y1,y1_blocks,X,dΩ) .< 1e-10)

############################################################################################

op = AffineFEOperator(biform,liform,X,Y)
block_op = AffineFEOperator(biform,liform,Xb,Yb)
