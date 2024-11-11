module BlockSparseMatrixAssemblersTests

using Test, LinearAlgebra, BlockArrays, SparseArrays

using Gridap
using Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField, Gridap.Algebra

using GridapDistributed
using PartitionedArrays
using GridapDistributed: BlockPVector, BlockPMatrix

get_edge_measures(Ω::Triangulation,dΩ) = sqrt∘CellField(get_array(∫(1)dΩ),Ω)
function get_edge_measures(Ω::GridapDistributed.DistributedTriangulation,dΩ)
  return sqrt∘CellField(map(get_array,local_views(∫(1)*dΩ)),Ω)
end

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
  yb = allocate_in_range(Ab)
  xb = allocate_in_domain(Ab); fill!(xb,1.0)
  mul!(yb,Ab,xb)

  y = allocate_in_range(A)
  x = allocate_in_domain(A); fill!(x,1.0)
  mul!(y,A,x)

  return is_same_vector(yb,y,Xb,X)
end

function _main(n_spaces,mfs,weakform,U,V)
  biform, liform = weakform

  # Normal assembly 
  Y = MultiFieldFESpace(fill(V,n_spaces))
  X = MultiFieldFESpace(fill(U,n_spaces))

  u = get_trial_fe_basis(X)
  v = get_fe_basis(Y)

  data = collect_cell_matrix_and_vector(X,Y,biform(u,v),liform(v))
  matdata = collect_cell_matrix(X,Y,biform(u,v))
  vecdata = collect_cell_vector(Y,liform(v))  

  assem = SparseMatrixAssembler(X,Y,LocallyAssembled())
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

  assem_blocks = SparseMatrixAssembler(Xb,Yb,LocallyAssembled())
  A1_blocks = assemble_matrix(assem_blocks,bmatdata);
  b1_blocks = assemble_vector(assem_blocks,bvecdata);
  @test is_same_vector(b1_blocks,b1,Yb,Y)
  @test is_same_matrix(A1_blocks,A1,Xb,X)

  assemble_matrix!(A1_blocks,assem_blocks,bmatdata);
  assemble_vector!(b1_blocks,assem_blocks,bvecdata);
  @test is_same_vector(b1_blocks,b1,Yb,Y)
  @test is_same_matrix(A1_blocks,A1,Xb,X)

  A2_blocks, b2_blocks = assemble_matrix_and_vector(assem_blocks,bdata)
  @test is_same_vector(b2_blocks,b2,Yb,Y)
  @test is_same_matrix(A2_blocks,A2,Xb,X)
  
  assemble_matrix_and_vector!(A2_blocks,b2_blocks,assem_blocks,bdata)
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

  # Conforming tests
  sol(x) = sum(x)
  reffe  = LagrangianRefFE(Float64,QUAD,1)
  V = FESpace(model, reffe; dirichlet_tags="boundary")
  U = TrialFESpace(sol,V)

  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  biform2((u1,u2),(v1,v2)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 + u1⋅v2)*dΩ
  liform2((v1,v2)) = ∫(v1 + v2)*dΩ
  biform3((u1,u2,u3),(v1,v2,v3)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 + u1⋅v2 + u3⋅v2 + u2⋅v3)*dΩ
  liform3((v1,v2,v3)) = ∫(v1 + v2 + v3)*dΩ

  for (n_spaces,weakform) in zip([2,3],[(biform2,liform2),(biform3,liform3)])
    for mfs in [BlockMultiFieldStyle(),BlockMultiFieldStyle(2,(1,n_spaces-1))]
      _main(n_spaces,mfs,weakform,U,V)
    end
  end

  # Non-conforming tests (tests whether we can fetch neighbors from non-ghosts)
  reffe = ReferenceFE(raviart_thomas,Float64,0)
  D = FESpace(model,reffe; dirichlet_tags="boundary")

  β_U = 100.0
  Γ = Boundary(model)
  Λ = Skeleton(model)
  n_Γ = get_normal_vector(Γ)
  n_Λ = get_normal_vector(Λ)
  dΓ = Measure(Γ,2)
  dΛ = Measure(Λ,2)
  h_e_Λ = get_edge_measures(Λ,dΛ)
  h_e_Γ = get_edge_measures(Γ,dΓ)
  lap_dg(u,v) = ∫(∇(u)⊙∇(v))dΩ -
                ∫(jump(v⊗n_Λ)⊙(mean(∇(u))))dΛ -
                ∫(mean(∇(v))⊙jump(u⊗n_Λ))dΛ -
                ∫(v⋅(∇(u)⋅n_Γ))dΓ -
                ∫((∇(v)⋅n_Γ)⋅u)dΓ +
                ∫(jump(v⊗n_Λ)⊙((β_U/h_e_Λ*jump(u⊗n_Λ))))dΛ +
                ∫(v⋅(β_U/h_e_Γ*u))dΓ

  f = VectorValue(1.0,1.0)
  biform4((u1,u2),(v1,v2)) = lap_dg(u1,v1) + lap_dg(u1,v1) + ∫(u2⋅v2)*dΩ
  liform4((v1,v2)) = ∫(v1⋅f)*dΩ
  _main(2,BlockMultiFieldStyle(),(biform4,liform4),D,D)
end

main(DebugArray, (2,2))

end