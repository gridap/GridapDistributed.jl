module HcurlProjectionTests

using Gridap
using PartitionedArrays
using GridapDistributed
using Test

u_ex_2D((x,y)) = 2*VectorValue(-y,x)
f_ex_2D(x) = u_ex_2D(x)
u_ex_3D((x,y,z)) = 2*VectorValue(-y,x,0.) - VectorValue(0.,-z,y)
f_ex_3D(x) = u_ex_3D(x)

function get_analytical_functions(Dc)
  if Dc==2
    return u_ex_2D, f_ex_2D
  else
    @assert Dc==3
    return u_ex_3D, f_ex_3D
  end
end

function solve_hcurl_projection(model::GridapDistributed.DistributedDiscreteModel{Dc},order) where {Dc}
  u_ex, f_ex=get_analytical_functions(Dc)

  V = FESpace(
    model, ReferenceFE(nedelec,order), conformity=:Hcurl, dirichlet_tags="boundary"
  )
  
  U = TrialFESpace(V,u_ex)
  
  trian = Triangulation(model)
  degree = 2*(order+1)
  dΩ = Measure(trian,degree)
      
  a(u,v) = ∫( (∇×u)⋅(∇×v) + u⋅v )dΩ
  l(v) = ∫(f_ex⋅v)dΩ

  op = AffineFEOperator(a,l,U,V)
  if (num_free_dofs(U)==0)
    # UMFPACK cannot handle empty linear systems
    uh = zero(U)
  else
    uh = solve(op)
  end
  uh,U
end 

function check_error_hcurl_projection(model::GridapDistributed.DistributedDiscreteModel{Dc},order,uh) where {Dc}
  trian = Triangulation(model)
  degree = 2*(order+1)
  dΩ = Measure(trian,degree)

  u_ex, f_ex = get_analytical_functions(Dc)
  
  eu = u_ex - uh

  l2(v) = sqrt(sum(∫(v⋅v)*dΩ))
  hcurl(v) = sqrt(sum(∫(v⋅v + (∇×v)⋅(∇×v))*dΩ))
  
  eu_l2 = l2(eu)
  eu_hcurl = hcurl(eu)
  
  tol = 1.0e-6
  @test eu_l2 < tol
  @test eu_hcurl < tol
end

function test_2d(ranks,parts,order)
  domain = (0,1,0,1)
  model = CartesianDiscreteModel(ranks,parts,domain,(4,4))
  solve_hcurl_projection(model,order) |> x -> check_error_hcurl_projection(model,order,x[1])
end 

function test_3d(ranks,parts,order)
  domain = (0,1,0,1,0,1)
  model = CartesianDiscreteModel(ranks,parts,domain,(4,4,4))
  solve_hcurl_projection(model,order) |> x -> check_error_hcurl_projection(model,order,x[1])
end 

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  
  if length(parts)==2 
    for order=0:2
      test_2d(ranks,parts,order)
    end
  elseif length(parts)==3
    for order=0:2
      test_3d(ranks,parts,order)
    end
  else 
      @assert false 
  end 
end

end #module
