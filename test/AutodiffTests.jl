module AutodiffTests

using Test
using Gridap, Gridap.Algebra
using GridapDistributed
using PartitionedArrays
using SparseArrays
using ForwardDiff
using Gridap.ODEs: TransientCellField, get_jacs

function main_sf(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,cells)

  u((x,y)) = (x+y)^k
  σ(∇u) = (1.0+∇u⋅∇u)*∇u
  dσ(∇du,∇u) = (2*∇u⋅∇du)*∇u + (1.0+∇u⋅∇u)*∇du
  f(x) = -divergence(y->σ(∇(u,y)),x)

  k = 1
  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*k)
  r(u,v) = ∫( ∇(v)⋅(σ∘∇(u)) - v*f )dΩ
  j(u,du,v) = ∫( ∇(v)⋅(dσ∘(∇(du),∇(u))) )dΩ

  op = FEOperator(r,j,U,V)
  op_AD = FEOperator(r,U,V)

  uh = interpolate(1.0,U)
  A = jacobian(op,uh)
  A_AD = jacobian(op_AD,uh)
  @test reduce(&,map(≈,partition(A),partition(A_AD)))

  g(v) = ∫(0.5*v⋅v)dΩ
  dg(v) = ∫(uh⋅v)dΩ
  b = assemble_vector(dg,U)
  b_AD = assemble_vector(gradient(g,uh),U)
  @test b ≈ b_AD

  # Skeleton AD
  # I would like to compare the results, but we cannot be using FD in parallel...
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,2*k)
  g_Λ(v) = ∫(mean(v))*dΛ
  r_Λ(u,v) = ∫(mean(u)*mean(v))*dΛ

  b_Λ_AD = assemble_vector(gradient(g_Λ,uh),U)
  A_Λ_AD = jacobian(FEOperator(r_Λ,U,V),uh)
end

function main_mf(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(4,4))

  k = 2
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
  reffe_p = ReferenceFE(lagrangian,Float64,k-1;space=:P)

  u(x) = VectorValue(x[2],-x[1])
  V = TestFESpace(model,reffe_u,dirichlet_tags="boundary")
  U = TrialFESpace(V,u)
  Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)

  X = MultiFieldFESpace([U,Q])
  Y = MultiFieldFESpace([V,Q])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+1))

  ν = 1.0
  f = VectorValue(0.0,0.0)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dc(u,du,dv) = ∫(dv⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  biform((du,dp),(dv,dq)) = ∫(ν*∇(dv)⊙∇(du) - (∇⋅dv)*dp - (∇⋅du)*dq)dΩ
  liform((dv,dq)) = ∫(dv⋅f)dΩ

  r((u,p),(v,q)) = biform((u,p),(v,q)) + c(u,v) - liform((v,q))
  j((u,p),(du,dp),(dv,dq)) = biform((du,dp),(dv,dq)) + dc(u,du,dv)

  op = FEOperator(r,j,X,Y)
  op_AD = FEOperator(r,X,Y)

  xh = interpolate([VectorValue(1.0,1.0),1.0],X)
  uh, ph = xh
  A = jacobian(op,xh)
  A_AD = jacobian(op_AD,xh)
  @test reduce(&,map(≈,partition(A),partition(A_AD)))

  g((v,q)) = ∫(0.5*v⋅v + 0.5*q*q)dΩ
  dg((v,q)) = ∫(uh⋅v + ph*q)dΩ
  b = assemble_vector(dg,X)
  b_AD = assemble_vector(gradient(g,xh),X)
  @test b ≈ b_AD
end

## MultiField AD with different triangulations for each field
function generate_trian(ranks,model,case)
  cell_ids = get_cell_gids(model)
  trians = map(ranks,local_views(model),partition(cell_ids)) do rank, model, ids
    cell_mask = zeros(Bool, num_cells(model))
    if case == :partial_trian
      if rank ∈ (1,2)
        cell_mask[own_to_local(ids)] .= true
      else
        t = own_to_local(ids)
        cell_mask[t[1:floor(Int,length(t)/2)]] .= true
      end
    elseif case == :half_empty_trian
      if rank ∈ (3,4)
        cell_mask[own_to_local(ids)] .= true
      end
    elseif case == :trian_with_empty_procs
      if rank ∈ (1,2)
        t = own_to_local(ids)
        cell_mask[t[1:floor(Int,length(t)/2)]] .= true
      end
    else
      error("Unknown case")
    end
    Triangulation(model,cell_mask)
  end
  GridapDistributed.DistributedTriangulation(trians,model)
end

function mf_different_fespace_trians(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(10,10))
  V2 = FESpace(model,ReferenceFE(lagrangian,VectorValue{2,Float64},1))
  V3 = FESpace(model,ReferenceFE(lagrangian,Float64,1))
  for case in (:boundary,:partial_trian, :half_empty_trian, :trian_with_empty_procs)
    if case == :boundary
      Γ = BoundaryTriangulation(model)
    else
      Γ = generate_trian(ranks,model,case)
    end
    dΓ = Measure(Γ,2)
    V1 = FESpace(Γ,ReferenceFE(lagrangian,Float64,1))
    X = MultiFieldFESpace([V1,V2,V3])
    uh = zero(X);

    f(xh) = ∫(xh[1]+xh[2]⋅xh[2]+xh[1]*xh[3])dΓ
    df(v,xh) = ∫(v[1]+2*v[2]⋅xh[2]+v[1]*xh[3]+xh[1]*v[3])dΓ
    du = gradient(f,uh)
    du_vec = assemble_vector(du,X)
    df_vec = assemble_vector(v->df(v,uh),X)

    @test df_vec ≈ du_vec

    f2(xh,yh) = ∫(xh[1]⋅yh[1]+xh[2]⋅yh[2]+xh[1]⋅xh[2]⋅yh[2]+xh[1]*xh[3]*yh[3])dΓ
    dv = get_fe_basis(X);
    j = jacobian(uh->f2(uh,dv),uh)
    J = assemble_matrix(j,X,X)

    f2_jac(xh,dxh,yh) = ∫(dxh[1]⋅yh[1]+dxh[2]⋅yh[2]+dxh[1]⋅xh[2]⋅yh[2]+xh[1]⋅dxh[2]⋅yh[2]+dxh[1]*xh[3]*yh[3]+xh[1]*dxh[3]*yh[3])dΓ
    op = FEOperator(f2,f2_jac,X,X)
    J_fwd = jacobian(op,uh)

    @test reduce(&,map(≈,partition(J),partition(J_fwd)))
  end
end

function skeleton_mf_different_fespace_trians(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(10,10))
  for case in (:partial_trian, :half_empty_trian, :trian_with_empty_procs)
    Γ = generate_trian(ranks,model,case)
    V1 = FESpace(Γ,ReferenceFE(lagrangian,Float64,1),conformity=:L2)
    V2 = FESpace(model,ReferenceFE(lagrangian,VectorValue{2,Float64},1),conformity=:L2)
    V3 = FESpace(model,ReferenceFE(lagrangian,Float64,1),conformity=:L2)
    X = MultiFieldFESpace([V1,V2,V3])
    uh = zero(X);
    Λ = SkeletonTriangulation(model)
    dΛ = Measure(Λ,2)

    f(xh) = ∫(mean(xh[1])+mean(xh[2])⋅mean(xh[2])+mean(xh[1])*mean(xh[3]))dΛ
    df(v,xh) = ∫(mean(v[1])+2*mean(v[2])⋅mean(xh[2])+mean(v[1])*mean(xh[3])+mean(xh[1])*mean(v[3]))dΛ
    du = gradient(f,uh)
    du_vec = assemble_vector(du,X)
    df_vec = assemble_vector(v->df(v,uh),X)

    @test df_vec ≈ du_vec

    # Skel jac
    f2(xh,yh) = ∫(mean(xh[1])⋅mean(yh[1])+mean(xh[2])⋅mean(yh[2])+mean(xh[1])⋅mean(xh[2])⋅mean(yh[2])+mean(xh[1])*mean(xh[3])*mean(yh[3]))dΛ
    dv = get_fe_basis(X);
    j = jacobian(uh->f2(uh,dv),uh);
    J = assemble_matrix(j,X,X)

    f2_jac(xh,dxh,yh) = ∫(mean(dxh[1])⋅mean(yh[1])+mean(dxh[2])⋅mean(yh[2])+mean(dxh[1])⋅mean(xh[2])⋅mean(yh[2]) +
      mean(xh[1])⋅mean(dxh[2])⋅mean(yh[2])+mean(dxh[1])*mean(xh[3])*mean(yh[3])+mean(xh[1])*mean(dxh[3])*mean(yh[3]))dΛ
    op = FEOperator(f2,f2_jac,X,X)
    J_fwd = jacobian(op,uh)

    @test reduce(&,map(≈,partition(J),partition(J_fwd)))
  end
end

function main_transient_sf(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,cells)

  u((x,y),t) = (x+y)^k + 2*t
  u(t::Real) = x -> u(x,t)
  σ(∇u) = (1.0+∇u⋅∇u)*∇u
  dσ(∇du,∇u) = (2*∇u⋅∇du)*∇u + (1.0+∇u⋅∇u)*∇du
  f(t) = x -> ∂t(u)(t)(x) - divergence(y->σ(∇(u(t),y)),x)

  k = 1
  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TransientTrialFESpace(V,u)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*k)
  r(t,u,v) = ∫( ∂t(u)⋅v + ∇(v)⋅(σ∘∇(u)) - v*f(t) )dΩ
  j_0(t,u,du,v) = ∫( ∇(v)⋅(dσ∘(∇(du),∇(u))) )dΩ
  j_t(t,u,dut,v) = ∫( dut⋅v )dΩ

  op = TransientFEOperator(r,(j_0,j_t),U,V)
  op_AD = TransientFEOperator(r,U,V)

  uh = interpolate(0.0,U(0.0))
  ∂tuₕ = interpolate(0.0,U(0.0))
  uhₜ = TransientCellField(uh,(∂tuₕ,))
  
  b = assemble_vector(v->r(1.0,uhₜ,v),V)
  A_0 = assemble_matrix((du,v)->j_0(1.0,uhₜ,du,v),U(1.0),V)
  A_t = assemble_matrix((dut,v)->j_t(1.0,uhₜ,dut,v),U(1.0),V)
  jac_0_AD = get_jacs(op_AD)[1]
  jac_t_AD = get_jacs(op_AD)[2]
  A_0_AD = assemble_matrix((du,v)->jac_0_AD(1.0,uhₜ,du,v),U(1.0),V)
  A_t_AD = assemble_matrix((dut,v)->jac_t_AD(1.0,uhₜ,dut,v),U(1.0),V)
  @test reduce(&,map(≈,partition(A_0),partition(A_0_AD)))
  @test reduce(&,map(≈,partition(A_t),partition(A_t_AD)))
end

function main(distribute,parts)
  main_sf(distribute,parts)
  main_mf(distribute,parts)
  mf_different_fespace_trians(distribute,parts)
  skeleton_mf_different_fespace_trians(distribute,parts)
  main_transient_sf(distribute,parts)
end

end