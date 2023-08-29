module BlockSparseMatrixAssemblersTests

using Test, LinearAlgebra, BlockArrays, SparseArrays

using Gridap
using Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField

using GridapDistributed
using PartitionedArrays
using GridapDistributed: BlockPVector, BlockPMatrix

function is_same_vector(x::BlockPVector,y::PVector,Ub,U)
  y_fespace = GridapDistributed.change_ghost(y,U.gids)
  x_fespace = GridapDistributed.change_ghost(x,Ub.gids)

  res = map(1:num_fields(Ub)) do i
    xi = restrict_to_field(Ub,x_fespace,i)
    yi = restrict_to_field(U,y_fespace,i)
    xi ≈ yi
  end
  return all(res)
end

function is_same_matrix(Ab::BlockPMatrix,A::PSparseMatrix,Xb,X)
  yb = mortar(map(Aii->pfill(0.0,partition(axes(Aii,1))),diag(blocks(Ab))));
  xb = mortar(map(Aii->pfill(1.0,partition(axes(Aii,2))),diag(blocks(Ab))));
  mul!(yb,Ab,xb)

  y = pfill(0.0,partition(axes(A,1)))
  x = pfill(1.0,partition(axes(A,2)))
  mul!(y,A,x)

  return is_same_vector(yb,y,Xb,X)
end

function _main(n_spaces,mfs,weakform,Ω,dΩ,U,V)
  biform, liform = weakform

  # Normal assembly 
  Y = MultiFieldFESpace(fill(V,n_spaces))
  X = MultiFieldFESpace(fill(U,n_spaces))

  u = get_trial_fe_basis(X)
  v = get_fe_basis(Y)

  data = collect_cell_matrix_and_vector(X,Y,biform(u,v),liform(v))
  matdata = collect_cell_matrix(X,Y,biform(u,v))
  vecdata = collect_cell_vector(Y,liform(v))  

  assem = SparseMatrixAssembler(X,Y,FullyAssembledRows())
  A1 = assemble_matrix(assem,matdata)
  b1 = assemble_vector(assem,vecdata)
  A2,b2 = assemble_matrix_and_vector(assem,data);

  # Block Assembly
  Yb  = MultiFieldFESpace(fill(V,n_spaces);style=mfs)
  Xb  = MultiFieldFESpace(fill(U,n_spaces);style=mfs)

  ub = get_trial_fe_basis(Xb)
  vb = get_fe_basis(Yb)

  bdata = collect_cell_matrix_and_vector(Xb,Yb,biform(ub,vb),liform(vb))
  bmatdata = collect_cell_matrix(Xb,Yb,biform(ub,vb))
  bvecdata = collect_cell_vector(Yb,liform(vb))

  assem_blocks = SparseMatrixAssembler(Xb,Yb,FullyAssembledRows())
  A1_blocks = assemble_matrix(assem_blocks,bmatdata);
  b1_blocks = assemble_vector(assem_blocks,bvecdata);
  @test is_same_vector(b1_blocks,b1,Yb,Y)
  @test is_same_matrix(A1_blocks,A1,Xb,X)

  A2_blocks, b2_blocks = assemble_matrix_and_vector(assem_blocks,bdata)
  @test is_same_vector(b2_blocks,b2,Yb,Y)
  @test is_same_matrix(A2_blocks,A2,Xb,X)

  op = AffineFEOperator(biform,liform,X,Y)
  block_op = AffineFEOperator(biform,liform,Xb,Yb)
  @test is_same_vector(get_vector(block_op),get_vector(op),Yb,Y)
  @test is_same_matrix(get_matrix(block_op),get_matrix(op),Xb,X)
end

############################################################################################

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(8,8))
  Ω = Triangulation(model)

  sol(x) = sum(x)
  reffe  = LagrangianRefFE(Float64,QUAD,1)
  V = FESpace(Ω, reffe; dirichlet_tags="boundary")
  U = TrialFESpace(sol,V)

  dΩ = Measure(Ω, 2)
  biform2((u1,u2),(v1,v2)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 + u1⋅v2)*dΩ
  liform2((v1,v2)) = ∫(v1 + v2)*dΩ
  biform3((u1,u2,u3),(v1,v2,v3)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 + u1⋅v2 + u3⋅v2 + u2⋅v3)*dΩ
  liform3((v1,v2,v3)) = ∫(v1 + v2 + v3)*dΩ

  for (n_spaces,weakform) in zip([2,3],[(biform2,liform2),(biform3,liform3)])
    for mfs in [BlockMultiFieldStyle(),BlockMultiFieldStyle(2,(1,n_spaces-1))]
      _main(n_spaces,mfs,weakform,Ω,dΩ,U,V)
    end
  end
end

end