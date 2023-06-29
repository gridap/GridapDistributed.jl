module DivConformingTests
using SparseMatricesCSR
import Gridap: ∇, divergence, DIV
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays
using Test
using FillArrays

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

  polytope=QUAD
  scalar_reffe=Gridap.ReferenceFEs.ReferenceFE(polytope,Gridap.ReferenceFEs.lagrangian,Float64,1)
  cell_types=collect(Fill(1,length(cell_vertex_lids)))
  cell_reffes=[scalar_reffe]
  grid = Gridap.Geometry.UnstructuredGrid(node_coordinates,
                                          cell_vertex_lids,
                                          cell_reffes,
                                          cell_types,
                                          Gridap.Geometry.NonOriented())
  Gridap.Geometry.UnstructuredDiscreteModel(grid)
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

  polytope=QUAD
  scalar_reffe=Gridap.ReferenceFEs.ReferenceFE(polytope,Gridap.ReferenceFEs.lagrangian,Float64,1)
  cell_types=collect(Fill(1,length(cell_vertex_lids)))
  cell_reffes=[scalar_reffe]
  grid = Gridap.Geometry.UnstructuredGrid(node_coordinates,
                                          cell_vertex_lids,
                                          cell_reffes,
                                          cell_types,
                                          Gridap.Geometry.NonOriented())
  Gridap.Geometry.UnstructuredDiscreteModel(grid)
end

function f(model,reffe)
    V = FESpace(model,reffe,conformity=:Hdiv)
    U = TrialFESpace(V)

    das = FullyAssembledRows()
    trian = Triangulation(das,model)
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

   models=map(ranks) do rank
      if (rank==1)
        setup_p1_model()
      else
        setup_p2_model()
      end
    end

    num_owners,owner_to_global,ghost_to_global,ghost_to_owner=map(ranks) do rank
      if (rank==1)
        1,[1],[2],Int32[2]
      else
        1,[2],[1],Int32[1]
      end
    end |> tuple_of_arrays

    indices=map(ranks, owner_to_global, ghost_to_global, ghost_to_owner) do rank, 
                                                                            owner_to_global,
                                                                            ghost_to_global,
                                                                            ghost_to_owner
      owner = rank
      ngdofs = 2
      own_indices=OwnIndices(ngdofs,owner,owner_to_global)
      ghost_indices=GhostIndices(ngdofs,ghost_to_global,ghost_to_owner)
      OwnAndGhostIndices(own_indices,ghost_indices)
    end
    gids=PRange(indices)

    model = GridapDistributed.DistributedDiscreteModel(models,gids)

    reffe=ReferenceFE(raviart_thomas,Float64,0)
    f(model,reffe)
    reffe=ReferenceFE(QUAD, raviart_thomas, 0)
    f(model,reffe)
  end

end # module
