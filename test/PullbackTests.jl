module PullbackTests
using SparseMatricesCSR

using Gridap
using Gridap.Geometry
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using GridapDistributed
using PartitionedArrays
using Test
using FillArrays

using Gridap: ∇, divergence, DIV

function setup_p1_model()
  ptr  = [ 1, 5, 9 ]
  data = [ 1,2,3,4, 2,5,4,6  ]
  cell_vertex_lids = Gridap.Arrays.Table(data,ptr)

  node_coordinates = Vector{Point{2,Float64}}(undef,6)
  node_coordinates[1]=Point{2,Float64}(0.0,0.0)
  node_coordinates[2]=Point{2,Float64}(0.5,0.0)
  node_coordinates[3]=Point{2,Float64}(0.0,1.0)
  node_coordinates[4]=Point{2,Float64}(0.5,1.0)
  node_coordinates[5]=Point{2,Float64}(1.0,0.0)
  node_coordinates[6]=Point{2,Float64}(1.0,1.0)

  reffe = ReferenceFEs.ReferenceFE(QUAD,ReferenceFEs.lagrangian,Float64,1)
  cell_types = collect(Fill(1,length(cell_vertex_lids)))
  grid = Geometry.UnstructuredGrid(
    node_coordinates, cell_vertex_lids, [reffe], cell_types, Geometry.NonOriented()
  )
  Geometry.UnstructuredDiscreteModel(grid)
end

function setup_p2_model()
  ptr  = [ 1, 5, 9 ]
  data = [ 1,2,3,4, 5,1,6,3  ]
  cell_vertex_lids = Gridap.Arrays.Table(data,ptr)

  node_coordinates = Vector{Point{2,Float64}}(undef,6)
  node_coordinates[1]=Point{2,Float64}(0.5,0.0)
  node_coordinates[2]=Point{2,Float64}(1.0,0.0)
  node_coordinates[3]=Point{2,Float64}(0.5,1.0)
  node_coordinates[4]=Point{2,Float64}(1.0,1.0)
  node_coordinates[5]=Point{2,Float64}(0.0,0.0)
  node_coordinates[6]=Point{2,Float64}(0.0,1.0)

  reffe = Gridap.ReferenceFEs.ReferenceFE(QUAD,ReferenceFEs.lagrangian,Float64,1)
  cell_types = collect(Fill(1,length(cell_vertex_lids)))
  grid = Geometry.UnstructuredGrid(
    node_coordinates, cell_vertex_lids, [reffe], cell_types, Geometry.NonOriented()
  )
  return Geometry.UnstructuredDiscreteModel(grid)
end

function setup_pλ_model(σ)
  # Two triangles sharing the diagonal of the unit square. `σ` relabels the four
  # vertices: same cells in the same slot order, different vertex ids.

  pts = [ Point(0.0,0.0), Point(1.0,0.0), Point(0.0,1.0), Point(1.0,1.0) ]
  cells = ( (1,2,3), (2,4,3) )

  node_coordinates = Vector{Point{2,Float64}}(undef,4)
  for old in eachindex(pts)
    node_coordinates[σ[old]] = pts[old]
  end
  data = reduce(vcat, [ [σ[v] for v in c] for c in cells ])
  cell_vertex_lids = Gridap.Arrays.Table(data, Int32[1,4,7])

  reffe = ReferenceFEs.ReferenceFE(TRI,ReferenceFEs.lagrangian,Float64,1)
  cell_types = collect(Fill(1,length(cell_vertex_lids)))
  grid = Geometry.UnstructuredGrid(
    node_coordinates, cell_vertex_lids, [reffe], cell_types, Geometry.NonOriented()
  )
  Geometry.UnstructuredDiscreteModel(grid)
end

# The PΛ elements order each face's dofs by that face's vertex ids, so the ids
# must be ones the ranks share. Here the ranks number the four vertices in
# opposite directions: both still produce the same `get_cell_dof_ids`, but the
# shape functions in the shared-edge slots differ by a sign, so an interpolant
# built from local ids is discontinuous across the diagonal.
function pλ_cross_rank(ranks)
  @assert length(ranks)==2

  models = map(ranks) do rank
    setup_pλ_model(rank == 1 ? [1,2,3,4] : [4,3,2,1])
  end
  cell_ids = map(ranks) do rank
    LocalIndices(2, rank, [1,2], Int32[1,2])
  end
  model = GridapDistributed.DistributedDiscreteModel(models, PRange(cell_ids))

  # No rank is locally sorted, so this mesh does exercise the non-trivial branch.
  local_allsorted = map(local_views(model)) do m
    all(issorted, Geometry.get_faces(Geometry.get_grid_topology(m), 2, 0))
  end
  @test !PartitionedArrays.getany(reduction(&, local_allsorted; init=true, destination=:all))

  u(x) = VectorValue(1.0, 0.0)
  Ω  = Triangulation(model)
  dΩ = Measure(Ω, 4)
  for name in (rotating_pλ, trimmed_pλ)
    V  = FESpace(model, ReferenceFE(name,Float64,1); conformity=CurlConformity())
    uh = interpolate(u, V)
    @test sqrt(sum(∫((uh-u)⋅(uh-u))dΩ)) < 1.0e-10
  end
end

function f(model,reffe,trian,das)
  V = FESpace(model,reffe)
  U = TrialFESpace(V)

  degree = 2
  dΩ = Measure(trian,degree)
  a(u,v) = ∫( u⋅v )*dΩ

  u  = get_trial_fe_basis(U)
  v  = get_fe_basis(V)
  dc = a(u,v)

  assem = SparseMatrixAssembler(U,V,das)
  data = collect_cell_matrix(U,V,a(u,v))
  A = assemble_matrix(assem,data)
  t1  = trian.trians.items[1]
  t2  = trian.trians.items[2]
  dc1 = dc.contribs.items[1]
  dc2 = dc.contribs.items[2]
  c1  = Gridap.CellData.get_contribution(dc1,t1)
  c2  = Gridap.CellData.get_contribution(dc2,t2)

  tol = 1.0e-12
  @test norm(c1[1]-c2[2]) < tol
  @test norm(c1[2]-c2[1]) < tol
end

function main(distribute,nranks)
  ranks = distribute(LinearIndices((prod(nranks),)))
  @assert isa(ranks,DebugArray)
  @assert length(ranks)==2

  models = map(ranks) do rank
    if (rank==1)
      setup_p1_model()
    else
      setup_p2_model()
    end
  end

  num_owners, owner_to_global, ghost_to_global, ghost_to_owner = map(ranks) do rank
    if (rank==1)
      1,[1],[2],Int32[2]
    else
      1,[2],[1],Int32[1]
    end
  end |> tuple_of_arrays

  indices = map(
    ranks, owner_to_global, ghost_to_global, ghost_to_owner
    ) do rank, owner_to_global, ghost_to_global, ghost_to_owner
    owner = rank
    ngdofs = 2
    own_indices=OwnIndices(ngdofs,owner,owner_to_global)
    ghost_indices=GhostIndices(ngdofs,ghost_to_global,ghost_to_owner)
    OwnAndGhostIndices(own_indices,ghost_indices)
  end
  gids = PRange(indices)

  model = GridapDistributed.DistributedDiscreteModel(models,gids)

  das = FullyAssembledRows()
  trian = Triangulation(das,model)

  reffe = ReferenceFE(raviart_thomas,Float64,0)
  f(model,reffe,trian,das)
  f(trian,reffe,trian,das)
  f(Triangulation(model),reffe,trian,das)

  reffe = ReferenceFE(QUAD, raviart_thomas, 0)
  f(model,reffe,trian,das)
  f(trian,reffe,trian,das)
  f(Triangulation(model),reffe,trian,das)

  reffe = ReferenceFE(nedelec,Float64,0)
  f(model,reffe,trian,das)
  f(trian,reffe,trian,das)
  f(Triangulation(model),reffe,trian,das)

  reffe = ReferenceFE(QUAD, nedelec, 0)
  f(model,reffe,trian,das)
  f(trian,reffe,trian,das)
  f(Triangulation(model),reffe,trian,das)

  pλ_cross_rank(ranks)
end

end # module
