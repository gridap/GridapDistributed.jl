using Pkg; Pkg.activate("./GridapDistributed");

using Test

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Geometry
using Gridap.CellData
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.MultiField

using GridapDistributed
using PartitionedArrays

function half_empty_trian(ranks,model)
  cell_ids = get_cell_gids(model)
  trians = map(ranks,local_views(model),partition(cell_ids)) do rank, model, ids
    cell_mask = zeros(Bool, num_cells(model))
    if rank ∈ (3,4)
      cell_mask[own_to_local(ids)] .= true
    end
    Triangulation(model,cell_mask)
  end
  GridapDistributed.DistributedTriangulation(trians,model)
end

np = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(5,5))
Ω = Triangulation(model)
# Γ = BoundaryTriangulation(model)
Γ = half_empty_trian(ranks,model)
dΩ = Measure(Ω,2)
dΓ = Measure(Γ,2)
V1 = FESpace(Γ,ReferenceFE(lagrangian,Float64,1))
V2 = FESpace(model,ReferenceFE(lagrangian,VectorValue{2,Float64},1))
V3 = FESpace(model,ReferenceFE(lagrangian,Float64,1))
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

# Skel
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