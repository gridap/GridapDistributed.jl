module MomentBasedFESpacesTests

# Distributed counterpart of Gridap's `NonOrientedMomentBasedTests`.

using Gridap
using Gridap.ReferenceFEs
using GridapDistributed
using PartitionedArrays
using Test

# a field of every space of order >= 1, and of order 0
linear(D) = x -> VectorValue(ntuple(i -> 1.0 + 2.0*x[i] - 0.5*sum(x), D))
constant(D) = x -> VectorValue(ntuple(i -> 1.0 - 0.5*i, D))

# Which orders of each family reproduce which interpolated field
const CASES = (
  # mesh    name              conformity         constant  linear
  (:ncube,  raviart_thomas,   DivConformity(),   (0,),     (1, 2)),
  (:ncube,  nedelec,          CurlConformity(),  (0,),     (1, 2)),
  (:ncube,  sdiv,             DivConformity(),   (),       (1, 2)),
  (:ncube,  scurl,            CurlConformity(),  (),       (1, 2)),
  (:ncube,  trimmed_sdiv,     DivConformity(),   (0,),     (1, 2)),
  (:ncube,  trimmed_scurl,    CurlConformity(),  (0,),     (1, 2)),
  (:simplex, raviart_thomas,  DivConformity(),   (0,),     (1, 2)),
  (:simplex, bdm,             DivConformity(),   (),       (1, 2)),
  (:simplex, rotating_pλ,     CurlConformity(),  (),       (1, 2)),
  (:simplex, trimmed_pλ,      CurlConformity(),  (1,),     (2,)),
)

function test_local_or_distributed()
  distributed = (
    (raviart_thomas, DivConformity()), (BDM(), DivConformity()),
    (sdiv, DivConformity()), (trimmed_sdiv, DivConformity()),
    (rotating_pλ, CurlConformity()), (trimmed_pλ, CurlConformity()),
  )
  local_only = (
    (nedelec, CurlConformity()), (scurl, CurlConformity()),
    (trimmed_scurl, CurlConformity()),          # covariant: face permutations only
    (MardalTaiWinther(), DivConformity()),      # contravariant, but permutation-based
    (raviart_thomas, L2Conformity()),           # no push-forward at all
    (Argyris(), DivConformity()), (Morley(), DivConformity()),
    (Hermite(), DivConformity()), (HellanHerrmannJohnson(), DivConformity()),
    (Regge(), CurlConformity()), (ArnoldWintherC(), DivConformity()),
    (ArnoldWintherNC(), DivConformity()),
    (GopalakrishnanLedererSchoberl(), L2Conformity()),
  )
  needs(name, conf) =
    GridapDistributed.needs_distributed_cell_bases_changes(name, Pushforward(name, conf))
  for (name, conf) in distributed
    @test needs(name, conf)
  end
  for (name, conf) in local_only
    @test !needs(name, conf)
  end
end

function interpolation_error(model, name, conformity, order, u)
  V = FESpace(model, ReferenceFE(name, Float64, order); conformity)
  uh = interpolate(u, V)
  dΩ = Measure(Triangulation(model), 2*(order + 2))
  return sqrt(sum(∫((uh - u)⋅(uh - u))dΩ))
end

# a contravariant factor of product reffe needs the distributed route even
# though the product's own push-forward is a `CartProdPushforward`.
function test_cartprod(model, p, order, tol)
  D = num_cell_dims(model)
  reffe = CartProdRefFE(ReferenceFE(p, raviart_thomas, order), 2)
  V = FESpace(model, reffe; conformity=CartProdConformity(DivConformity(), 2))

  u = x -> TensorValue{D,2}(ntuple(i -> 1.0 + 2.0*x[mod1(i, D)] - 0.5*sum(x), 2*D))
  uh = interpolate(u, V)
  dΩ = Measure(Triangulation(model), 2*(order + 2))
  @test sqrt(sum(∫((uh - u)⊙(uh - u))dΩ)) < tol
end

function main(distribute, parts)
  test_local_or_distributed()

  # The orders in CASES were measured on hexes and tets; every caller passes a
  # 3-tuple, the MPI driver by appending a 1 to its 2-tuple.
  D = length(parts)
  @assert D == 3
  ranks = distribute(LinearIndices((prod(parts),)))
  domain = ntuple(i -> isodd(i) ? 0 : 1, 2*D)

  ncube = CartesianDiscreteModel(ranks, parts, domain, ntuple(_ -> 2, D))
  models = (; ncube, simplex = simplexify(ncube))

  tol = 1.0e-10
  for (mesh, name, conformity, const_orders, linear_orders) in CASES
    for (orders, u) in ((const_orders, constant(D)), (linear_orders, linear(D)))
      for order in orders
        @test interpolation_error(models[mesh], name, conformity, order, u) < tol
      end
    end
  end

  test_cartprod(ncube, HEX, 1, tol)
end

end # module
