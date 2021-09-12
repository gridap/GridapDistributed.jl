const P4EST_2_GRIDAP_FACE_2D  = [ 3, 4, 1, 2 ]
const GRIDAP_2_P4EST_FACE_2D  = [ 3, 4, 1, 2 ]

p4est_wrapper.quadrant_data(x::Clong) = reinterpret(p4est_wrapper.quadrant_data, x)

function p4est_get_quadrant_vertex_coordinates(connectivity::Ptr{p4est_connectivity_t},
                                               treeid::p4est_topidx_t,
                                               x::p4est_qcoord_t,
                                               y::p4est_qcoord_t,
                                               level::Int8,
                                               corner::Cint,
                                               vxy::Ptr{Cdouble})

    myself=Ref{p4est_quadrant_t}(
      p4est_quadrant_t(x,y,level,Int8(0),Int16(0),
                       p4est_wrapper.quadrant_data(Clong(0))))
    neighbour=Ref{p4est_quadrant_t}(myself[])
    if corner == 1
       p4est_quadrant_face_neighbor(myself,corner,neighbour)
    elseif corner == 2
       p4est_quadrant_face_neighbor(myself,corner+1,neighbour)
    elseif corner == 3
       p4est_quadrant_corner_neighbor(myself,corner,neighbour)
    end
    # Extract numerical coordinates of lower_left
    # corner of my corner neighbour
    p4est_qcoord_to_vertex(connectivity,
                           treeid,
                           neighbour[].x,
                           neighbour[].y,
                           vxy)
end

function UniformlyRefinedForestOfOctreesDiscreteModel(comm::Communicator,
                                                      coarse_discrete_model::DiscreteModel,
                                                      num_uniform_refinements::Int)


  mpicomm = comm.comm #p4est_wrapper.P4EST_ENABLE_MPI ? MPI.COMM_WORLD : Cint(0)

  # Create a connectivity structure for the unit square.
  unitsquare_connectivity = p4est_connectivity_new_unitsquare()
  @assert unitsquare_connectivity != C_NULL

  # Create a new forest
  unitsquare_forest = p4est_new_ext(mpicomm,
                                    unitsquare_connectivity,
                                    Cint(0), Cint(num_uniform_refinements), Cint(1), Cint(0),
                                    C_NULL, C_NULL)
  @assert unitsquare_forest != C_NULL

  unitsquare_ghost=p4est_ghost_new(unitsquare_forest,p4est_wrapper.P4EST_CONNECT_FULL)
  @assert unitsquare_ghost != C_NULL

  # Build the ghost layer.
  p4est_ghost = unitsquare_ghost[]
  p4est       = unitsquare_forest[]

  # Obtain ghost quadrants
  ptr_ghost_quadrants = Ptr{p4est_quadrant_t}(p4est_ghost.ghosts.array)
  proc_offsets = unsafe_wrap(Array, p4est_ghost.proc_offsets, p4est_ghost.mpisize+1)

  global_first_quadrant = unsafe_wrap(Array,
                                      p4est.global_first_quadrant,
                                      p4est.mpisize+1)

  n = p4est.global_num_quadrants
  cellindices = DistributedIndexSet(comm,n) do part
    lid_to_gid   = Vector{Int}(undef, p4est.local_num_quadrants + p4est_ghost.ghosts.elem_count)
    lid_to_part  = Vector{Int}(undef, p4est.local_num_quadrants + p4est_ghost.ghosts.elem_count)

    for i=1:p4est.local_num_quadrants
      lid_to_gid[i]  = global_first_quadrant[MPI.Comm_rank(mpicomm)+1]+i
      lid_to_part[i] = MPI.Comm_rank(mpicomm)+1
    end

    k=p4est.local_num_quadrants+1
    for i=1:p4est_ghost.mpisize
      for j=proc_offsets[i]:proc_offsets[i+1]-1
        quadrant       = ptr_ghost_quadrants[j+1]
        piggy3         = quadrant.p.piggy3
        lid_to_part[k] = i
        lid_to_gid[k]  = global_first_quadrant[i]+piggy3.local_num+1
        k=k+1
      end
    end
    # if (part==2)
    #    print(n,"\n")
    #    print(lid_to_gid,"\n")
    #    print(lid_to_part,"\n")
    # end
    IndexSet(n,lid_to_gid,lid_to_part)
  end

  ptr_unitsquare_lnodes=p4est_lnodes_new(unitsquare_forest, unitsquare_ghost, Cint(1))
  @assert ptr_unitsquare_lnodes != C_NULL

  unitsquare_lnodes=ptr_unitsquare_lnodes[]

  nvertices = unitsquare_lnodes.vnodes
  element_nodes = unsafe_wrap(Array,
                              unitsquare_lnodes.element_nodes,
                              unitsquare_lnodes.num_local_elements*nvertices)

  nonlocal_nodes = unsafe_wrap(Array,
                               unitsquare_lnodes.nonlocal_nodes,
                               unitsquare_lnodes.num_local_nodes-unitsquare_lnodes.owned_count)

  k = 1
  cell_vertex_gids=DistributedVector(cellindices,cellindices) do part, indices
    n = length(indices.lid_to_owner)
    ptrs = Vector{Int}(undef,n+1)
    ptrs[1]=1
    for i=1:n
      ptrs[i+1]=ptrs[i]+nvertices
    end
    k=1
    current=1
    data = Vector{Int}(undef, ptrs[n+1]-1)
    for i=1:unitsquare_lnodes.num_local_elements
      for j=1:nvertices
        l=element_nodes[k+j-1]
        if (l < unitsquare_lnodes.owned_count)
          data[current]=unitsquare_lnodes.global_offset+l+1
        else
          data[current]=nonlocal_nodes[l-unitsquare_lnodes.owned_count+1]+1
        end
        current=current+1
      end
      k=k+nvertices
    end
    Gridap.Arrays.Table(data,ptrs)
  end
  exchange!(cell_vertex_gids)

  cell_vertex_lids_nlvertices=DistributedData(cell_vertex_gids) do part, cell_vertex_gids
    g2l=Dict{Int,Int}()
    current=1
    data=Vector{Int}(undef,length(cell_vertex_gids.data))
    for (i,gid) in enumerate(cell_vertex_gids.data)
      if haskey(g2l,gid)
        data[i]=g2l[gid]
      else
        data[i]=current
        g2l[gid]=current
        current=current+1
      end
    end
    (Gridap.Arrays.Table(data,cell_vertex_gids.ptrs), current-1)
  end


  tree_offsets = unsafe_wrap(Array, p4est_ghost.tree_offsets, p4est_ghost.num_trees+1)
  D=2
  dnode_coordinates=DistributedData(cell_vertex_lids_nlvertices) do part, (cell_vertex_lids, nl)
     node_coordinates=Vector{Point{D,Float64}}(undef,nl)
     current=1
     num_trees=p4est.last_local_tree-p4est.first_local_tree+1
     vxy=Vector{Cdouble}(undef,D)
     pvxy=pointer(vxy,1)
     cell_lids=cell_vertex_lids.data
     for itree=1:num_trees
       tree = p4est_tree_array_index(p4est.trees, itree-1)[]
       for cell=1:tree.quadrants.elem_count
          quadrant=p4est_quadrant_array_index(tree.quadrants, cell-1)[]
          for vertex=1:nvertices
             p4est_get_quadrant_vertex_coordinates(unitsquare_connectivity,
                                               p4est_topidx_t(itree-1),
                                               quadrant.x,
                                               quadrant.y,
                                               quadrant.level,
                                               Cint(vertex-1),
                                               pvxy)
            #  if (MPI.Comm_rank(comm.comm)==0)
            #     println(vxy)
            #  end
             node_coordinates[cell_lids[current]]=Point{D,Float64}(vxy...)
             current=current+1
          end
       end
     end

     # Go over ghost cells
     for i=1:p4est_ghost.num_trees
      for j=tree_offsets[i]:tree_offsets[i+1]-1
          quadrant = ptr_ghost_quadrants[j+1]
          for vertex=1:nvertices
            p4est_get_quadrant_vertex_coordinates(unitsquare_connectivity,
                                              p4est_topidx_t(i-1),
                                              quadrant.x,
                                              quadrant.y,
                                              quadrant.level,
                                              Cint(vertex-1),
                                              pvxy)
          #  if (MPI.Comm_rank(comm.comm)==0)
          #     println(vxy)
          #  end
           node_coordinates[cell_lids[current]]=Point{D,Float64}(vxy...)
           current=current+1
         end
       end
     end
     node_coordinates
  end

  dgrid_and_topology=DistributedData(cell_vertex_lids_nlvertices,dnode_coordinates) do part, (cell_vertex_lids, nl), node_coordinates
      scalar_reffe=Gridap.ReferenceFEs.ReferenceFE(QUAD,Gridap.ReferenceFEs.lagrangian,Float64,1)
      cell_types=collect(Fill(1,length(cell_vertex_lids)))
      cell_reffes=[scalar_reffe]
      grid = Gridap.Geometry.UnstructuredGrid(node_coordinates,
                                              cell_vertex_lids,
                                              cell_reffes,
                                              cell_types,
                                              Gridap.Geometry.Oriented())

      topology = Gridap.Geometry.UnstructuredGridTopology(node_coordinates,
                                        cell_vertex_lids,
                                        cell_types,
                                        map(Gridap.ReferenceFEs.get_polytope, cell_reffes),
                                        Gridap.Geometry.Oriented())
      #writevtk(grid,"grid$(part)")
      grid,topology
  end


  coarse_grid_topology  = Gridap.Geometry.get_grid_topology(coarse_discrete_model)
  coarse_grid_labeling  = Gridap.Geometry.get_face_labeling(coarse_discrete_model)


  coarse_cell_vertices = Gridap.Geometry.get_faces(coarse_grid_topology,D,0)
  coarse_cell_faces    = Gridap.Geometry.get_faces(coarse_grid_topology,D,1)


  num_trees=p4est.last_local_tree-p4est.first_local_tree+1
  owned_trees_offset=Vector{Int}(undef,num_trees+1)
  owned_trees_offset[1]=1
  for itree=1:num_trees
    tree = p4est_tree_array_index(p4est.trees, itree-1)[]
    owned_trees_offset[itree+1]=owned_trees_offset[itree]+tree.quadrants.elem_count
  end
  # if (MPI.Comm_rank(comm.comm)==0)
  #   println("ZZZZZZZZ ",owned_trees_offset)
  # end

  dface_labeling=DistributedData(dgrid_and_topology) do part, (grid,topology)
     # Iterate over corners
     num_vertices=Gridap.Geometry.num_faces(topology,0)
     vertex_to_entity=zeros(Int,num_vertices)
     cell_vertices=Gridap.Geometry.get_faces(topology,D,0)

     # Corner iterator callback
     function jcorner_callback(pinfo   :: Ptr{p4est_iter_corner_info_t},
                               user_data :: Ptr{Cvoid})
        info=pinfo[]
        sides=Ptr{p4est_iter_corner_side_t}(info.sides.array)
        nsides=info.sides.elem_count
        # We are on the interior of a tree
        data=sides[1]
        if data.is_ghost==1
           ref_cell=p4est.local_num_quadrants+data.quadid+1
        else
           ref_cell=data.quadid+1
        end
        tree=sides[1].treeid+1
        corner=sides[1].corner+1
        ref_cornergid=cell_vertices[ref_cell][corner]
        # if (MPI.Comm_rank(comm.comm)==0)
        #   println("XXX ", ref_cell, " ", ref_cornergid, " ", info.tree_boundary)
        # end
        if (info.tree_boundary!=0)
          if (info.tree_boundary == p4est_wrapper.P4EST_CONNECT_CORNER)
              # The current corner is also a corner of the coarse mesh
              coarse_cornergid=coarse_cell_vertices[tree][corner]
              vertex_to_entity[ref_cornergid]=
                 coarse_grid_labeling.d_to_dface_to_entity[1][coarse_cornergid]
          end
        else
          # We are on the interior of a tree
          vertex_to_entity[ref_cornergid]=coarse_grid_labeling.d_to_dface_to_entity[3][tree]
        end
        # if (MPI.Comm_rank(comm.comm)==0)
        #   println("YYY ", cell_vertices)
        # end
        nothing
     end

     #  C-callable face callback
     ccorner_callback = @cfunction($jcorner_callback,
                                   Cvoid,
                                   (Ptr{p4est_iter_corner_info_t},Ptr{Cvoid}))

     # Iterate over faces
     num_faces=Gridap.Geometry.num_faces(topology,1)
     face_to_entity=zeros(Int,num_faces)
     cell_faces=Gridap.Geometry.get_faces(topology,D,1)

     # Face iterator callback
     function jface_callback(pinfo     :: Ptr{p4est_iter_face_info_t},
                             user_data :: Ptr{Cvoid})
        info=pinfo[]
        sides=Ptr{p4est_iter_face_side_t}(info.sides.array)
        nsides=info.sides.elem_count
        # We are on the interior of a tree
        data=sides[1].is.full
        if data.is_ghost==1
           ref_cell=p4est.local_num_quadrants+data.quadid+1
        else
           ref_cell=data.quadid+1
        end
        tree=sides[1].treeid+1
        face=sides[1].face+1
        gridap_face=P4EST_2_GRIDAP_FACE_2D[face]

        poly_faces=Gridap.ReferenceFEs.get_faces(QUAD)
        poly_face_range=Gridap.ReferenceFEs.get_dimrange(QUAD,1)
        poly_first_face=first(poly_face_range)
        poly_face=poly_first_face+gridap_face-1

        # if (MPI.Comm_rank(comm.comm)==0)
        #   println("XXX ", ref_cell, " ", ref_facegid, " ", info.tree_boundary)
        # end
        if (info.tree_boundary!=0)
          coarse_facegid=coarse_cell_faces[tree][gridap_face]
          # We are on the boundary of coarse mesh or inter-octree boundary
          for poly_incident_face in poly_faces[poly_face]
            if poly_incident_face == poly_face
              ref_facegid=cell_faces[ref_cell][gridap_face]
              face_to_entity[ref_facegid]=
                coarse_grid_labeling.d_to_dface_to_entity[2][coarse_facegid]
            else
              ref_cornergid=cell_vertices[ref_cell][poly_incident_face]
              vertex_to_entity[ref_cornergid]=
                 coarse_grid_labeling.d_to_dface_to_entity[2][coarse_facegid]
            end
          end
        else
          # We are on the interior of a tree
          ref_facegid=cell_faces[ref_cell][gridap_face]
          face_to_entity[ref_facegid]=coarse_grid_labeling.d_to_dface_to_entity[3][tree]
        end
        nothing
     end

    #  C-callable face callback
    cface_callback = @cfunction($jface_callback,
                                 Cvoid,
                                 (Ptr{p4est_iter_face_info_t},Ptr{Cvoid}))


    # Iterate over cells
    num_cells=Gridap.Geometry.num_faces(topology,2)
    cell_to_entity=zeros(Int,num_cells)

    # Face iterator callback
    function jcell_callback(pinfo     :: Ptr{p4est_iter_volume_info_t},
                            user_data :: Ptr{Cvoid})
      info=pinfo[]
      tree=info.treeid+1
      cell=info.quadid+1
      cell_to_entity[cell]=coarse_grid_labeling.d_to_dface_to_entity[3][tree]
      nothing
    end
    ccell_callback = @cfunction($jcell_callback,
                                 Cvoid,
                                 (Ptr{p4est_iter_volume_info_t},Ptr{Cvoid}))

    p4est_iterate(unitsquare_forest,unitsquare_ghost,C_NULL,C_NULL,cface_callback,C_NULL)
    p4est_iterate(unitsquare_forest,unitsquare_ghost,C_NULL,ccell_callback,C_NULL,ccorner_callback)

     #  struct FaceLabeling <: GridapType
     #   d_to_dface_to_entity::Vector{Vector{Int32}}
     #   tag_to_entities::Vector{Vector{Int32}}
     #   tag_to_name::Vector{String}
     # end
    #  if (MPI.Comm_rank(comm.comm)==0)
    #    println("XXX ", vertex_to_entity)
    #    println("XXX ", face_to_entity)
    #    println("XXX ", cell_to_entity)
    d_to_dface_to_entity    = Vector{Vector{Int}}(undef,3)
    d_to_dface_to_entity[1] = vertex_to_entity
    d_to_dface_to_entity[2] = face_to_entity
    d_to_dface_to_entity[3] = cell_to_entity

    Gridap.Geometry.FaceLabeling(d_to_dface_to_entity,
               coarse_grid_labeling.tag_to_entities,
               coarse_grid_labeling.tag_to_name)
 end

  ddiscretemodel=
    DistributedData(comm,dgrid_and_topology,dface_labeling) do part, (grid,topology), face_labeling
      Gridap.Geometry.UnstructuredDiscreteModel(grid,topology,face_labeling)
    end

  # Write forest to VTK file
  #p4est_vtk_write_file(unitsquare_forest, C_NULL, "my_step")

  # Destroy lnodes
  p4est_lnodes_destroy(ptr_unitsquare_lnodes)
  # Destroy ghost
  p4est_ghost_destroy(unitsquare_ghost)
  # Destroy the forest
  p4est_destroy(unitsquare_forest)
  # Destroy the connectivity
  p4est_connectivity_destroy(unitsquare_connectivity)

  DistributedDiscreteModel(ddiscretemodel,cellindices)

end
