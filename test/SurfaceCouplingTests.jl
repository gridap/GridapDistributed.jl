module SurfaceCouplingTests

using Gridap
import Gridap: ∇
using GridapDistributed
using PartitionedArrays
using Test

# Analytical functions
u(x) = VectorValue( x[1]^2 + 2*x[2]^2, -x[1]^2 )
∇u(x) = TensorValue( 2*x[1], 4*x[2], -2*x[1], zero(x[1]) )
Δu(x) = VectorValue( 6, -2 )
p(x) = x[1] + 3*x[2]
∇p(x) = VectorValue(1,3)
s(x) = -Δu(x)
f(x) = -Δu(x) + ∇p(x)
g(x) = tr(∇u(x))
∇(::typeof(u)) = ∇u
∇(::typeof(p)) = ∇p

function main(parts)

  # Mesh
  parts = get_part_ids(SequentialBackend(),(2,2))
  cells = (10,10)
  domain = (0,1,0,1)
  model = CartesianDiscreteModel(parts,domain,cells)
  function is_in(coords)
    R = 0.4
    n = length(coords)
    x = (1/n)*sum(coords) - Point(0.5,0.5)
    d = x[1]^2 + x[2]^2 - R^2
    d < 0
  end
  cell_to_entity = map_parts(local_views(model)) do model
    grid = get_grid(model)
    cell_to_coords = get_cell_coordinates(grid)
    cell_to_is_solid = lazy_map(is_in,cell_to_coords)
    cell_to_is_fluid = lazy_map(!,cell_to_is_solid)
    labels = get_face_labeling(model)
    cell_to_entity = labels.d_to_dface_to_entity[end]
    solid = maximum(cell_to_entity) + 1
    fluid = solid + 1
    cell_to_entity[Gridap.Arrays.collect1d(cell_to_is_solid)] .= solid
    cell_to_entity[Gridap.Arrays.collect1d(cell_to_is_fluid)] .= fluid
    add_tag!(labels,"solid",[solid])
    add_tag!(labels,"fluid",[fluid])
    add_tag_from_tags!(labels,"dirichlet",[1,2,5])
    add_tag_from_tags!(labels,"neumann",[6,7,8])
    cell_to_entity
  end
  cell_gids=get_cell_gids(model)
  exchange!(cell_to_entity,cell_gids.exchanger) # Make tags consistent

  # Domains and measures
  Ω = Interior(model)
  Ωs = Interior(model,tags="solid")
  Ωf = Interior(model,tags="fluid")
  Λ = Boundary(model,tags="neumann")
  Γ = Interface(Ωf,Ωs)
  n_Λ = get_normal_vector(Λ)
  n_Γ = get_normal_vector(Γ)
  k = 2
  degree = 2*k
  dΩ = Measure(Ω,degree)
  dΩs = Measure(Ωs,degree)
  dΩf = Measure(Ωf,degree)
  dΛ = Measure(Λ,degree)
  dΓ = Measure(Γ,degree)

  # FE Spaces
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
  reffe_p = ReferenceFE(lagrangian,Float64,k-1,space=:P)
  V = TestFESpace(Ω,reffe_u,dirichlet_tags="dirichlet")
  Q = TestFESpace(Ωf,reffe_p)
  U = TrialFESpace(V,u)
  P = Q
  Y = MultiFieldFESpace([V,Q])
  X = MultiFieldFESpace([U,P])

  # Weak form
  a((u,p),(v,q)) =
    ∫( ∇(v)⊙∇(u) )*dΩs +
    ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )*dΩf

  l((v,q)) =
    ∫( v⋅s )*dΩs +
    ∫( v⋅f + q*g )*dΩf +
    ∫( v⋅(n_Λ⋅∇u) - (n_Λ⋅v)*p )*dΛ +
    ∫( - (n_Γ.⁺⋅v.⁺)*p )*dΓ

  # FE problem
  op = AffineFEOperator(a,l,X,Y)
  uh, ph = solve(op)

  # Errors
  eu = u - uh
  ep = p - ph
  eu_l2 = sqrt(sum(∫( eu⋅eu )*dΩ))
  eu_h1 = sqrt(sum(∫( eu⋅eu + ∇(eu)⊙∇(eu) )*dΩ))
  ep_l2 = sqrt(sum(∫( ep*ep )*dΩf))
  tol = 1.0e-9
  @test eu_l2 < tol
  @test eu_h1 < tol
  @test ep_l2 < tol

end

end # module
