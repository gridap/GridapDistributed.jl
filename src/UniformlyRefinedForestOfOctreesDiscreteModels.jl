const P4EST_2_GRIDAP_FACET_2D  = [ 3, 4, 1, 2 ]
const GRIDAP_2_P4EST_FACET_2D  = [ 3, 4, 1, 2 ]


const P4EST_2_GRIDAP_FACET_3D  = [ 5, 6, 3, 4, 1, 2 ]
const GRIDAP_2_P4EST_FACET_3D  = [ 5, 6, 3, 4, 1, 2 ]

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


function  p8est_get_quadrant_vertex_coordinates(connectivity::Ptr{p8est_connectivity_t},
                                                treeid::p4est_topidx_t,
                                                x::p4est_qcoord_t,
                                                y::p4est_qcoord_t,
                                                z::p4est_qcoord_t,
                                                level::Int8,
                                                corner::Cint,
                                                vxyz::Ptr{Cdouble})

  myself=Ref{p8est_quadrant_t}(
       p8est_quadrant_t(x,y,z,level,Int8(0),Int16(0),
                        p4est_wrapper.quadrant_data(Clong(0))))
  neighbour=Ref{p8est_quadrant_t}(myself[])

  if ( corner == 1 )
    p8est_quadrant_face_neighbor(myself,Cint(1),neighbour)
  elseif ( corner == 2 )
    p8est_quadrant_face_neighbor(myself,Cint(3),neighbour)
  elseif ( corner == 3 )
    p8est_quadrant_edge_neighbor(myself,Cint(11),neighbour)
  elseif ( corner == 4 )
    p8est_quadrant_face_neighbor(myself,Cint(5),neighbour)
  elseif ( corner == 5 )
    p8est_quadrant_edge_neighbor(myself,Cint(7),neighbour)
  elseif ( corner == 6 )
    p8est_quadrant_edge_neighbor(myself,Cint(3),neighbour)
  elseif ( corner == 7 )
    p8est_quadrant_corner_neighbor(myself,Cint(7),neighbour)
  end

  # Extract numerical coordinates of lower_left corner of my corner neighbour
  p8est_qcoord_to_vertex(connectivity,
                         treeid,
                         neighbour[].x,
                         neighbour[].y,
                         neighbour[].z,
                         vxyz)
end


function setup_pXest_connectivity(
  coarse_discrete_model::DiscreteModel{Dc,Dp}) where {Dc,Dp}

  trian=Triangulation(coarse_discrete_model)
  node_coordinates=Gridap.Geometry.get_node_coordinates(trian)
  cell_nodes_ids=Gridap.Geometry.get_cell_node_ids(trian)

  if (Dc==2)
    pconn=p4est_connectivity_new(
        p4est_topidx_t(length(node_coordinates)),         # num_vertices
        p4est_topidx_t(num_cells(coarse_discrete_model)), # num_trees
        p4est_topidx_t(0),
        p4est_topidx_t(0))
  else
    @assert Dc==3
    pconn=p8est_connectivity_new(
        p4est_topidx_t(length(node_coordinates)),         # num_vertices
        p4est_topidx_t(num_cells(coarse_discrete_model)), # num_trees
        p4est_topidx_t(0),
        p4est_topidx_t(0),
        p4est_topidx_t(0),
        p4est_topidx_t(0))
  end

  conn=pconn[]
  vertices=unsafe_wrap(Array, conn.vertices, length(node_coordinates)*3)
  current=1
  for i=1:length(node_coordinates)
    p=node_coordinates[i]
    for j=1:Dp
      vertices[current]=Cdouble(p[j])
      current=current+1
    end
    if (Dp==2)
      vertices[current]=Cdouble(0.0) # Z coordinate always to 0.0 in 2D
      current=current+1
    end
  end
  #print("XXX", vertices, "\n")

  tree_to_vertex=unsafe_wrap(Array, conn.tree_to_vertex, length(cell_nodes_ids)*(2^Dc))
  c=Gridap.Arrays.array_cache(cell_nodes_ids)
  current=1
  for j=1:length(cell_nodes_ids)
     ids=Gridap.Arrays.getindex!(c,cell_nodes_ids,j)
     for id in ids
      tree_to_vertex[current]=p4est_topidx_t(id-1)
      current=current+1
     end
  end


  # /*
  #  * Fill tree_to_tree and tree_to_face to make sure we have a valid
  #  * connectivity.
  #  */
  PXEST_FACES=2*Dc
  tree_to_tree=unsafe_wrap(Array, conn.tree_to_tree, conn.num_trees*PXEST_FACES )
  tree_to_face=unsafe_wrap(Array, conn.tree_to_face, conn.num_trees*PXEST_FACES )
  for tree=1:conn.num_trees
    for face=1:PXEST_FACES
      tree_to_tree[PXEST_FACES * (tree-1) + face] = tree-1
      tree_to_face[PXEST_FACES * (tree-1) + face] = face-1
    end
  end

  if (Dc==2)
    p4est_connectivity_complete(pconn)
    @assert Bool(p4est_connectivity_is_valid(pconn))
  else
    p8est_connectivity_complete(pconn)
    @assert Bool(p8est_connectivity_is_valid(pconn))
  end
  pconn
end

function setup_pXest(::Type{Val{Dc}}, comm, connectivity, num_uniform_refinements) where Dc
   if (Dc==2)
       p4est_new_ext(comm.comm,
                     connectivity,
                     Cint(0), Cint(num_uniform_refinements), Cint(1), Cint(0),
                     C_NULL, C_NULL)
   else
      p8est_new_ext(comm.comm,
                    connectivity,
                    Cint(0), Cint(num_uniform_refinements), Cint(1), Cint(0),
                    C_NULL, C_NULL)
   end
end

function setup_pXest_ghost(::Type{Val{Dc}}, ptr_pXest) where Dc
  if (Dc==2)
    p4est_ghost_new(ptr_pXest,p4est_wrapper.P4EST_CONNECT_FULL)
  else
    p8est_ghost_new(ptr_pXest,p4est_wrapper.P8EST_CONNECT_FULL)
  end
end

function setup_cell_indexset(::Type{Val{Dc}}, comm, ptr_pXest, ptr_pXest_ghost) where Dc
  pXest_ghost = ptr_pXest_ghost[]
  pXest       = ptr_pXest[]

  # Obtain ghost quadrants
  if (Dc==2)
    ptr_ghost_quadrants = Ptr{p4est_quadrant_t}(pXest_ghost.ghosts.array)
  else
    ptr_ghost_quadrants = Ptr{p8est_quadrant_t}(pXest_ghost.ghosts.array)
  end
  proc_offsets = unsafe_wrap(Array, pXest_ghost.proc_offsets, pXest_ghost.mpisize+1)

  global_first_quadrant = unsafe_wrap(Array,
                                      pXest.global_first_quadrant,
                                      pXest.mpisize+1)

  n = pXest.global_num_quadrants
  cellindices = DistributedIndexSet(comm,n) do part
    lid_to_gid   = Vector{Int}(undef, pXest.local_num_quadrants + pXest_ghost.ghosts.elem_count)
    lid_to_part  = Vector{Int}(undef, pXest.local_num_quadrants + pXest_ghost.ghosts.elem_count)

    for i=1:pXest.local_num_quadrants
      lid_to_gid[i]  = global_first_quadrant[MPI.Comm_rank(comm.comm)+1]+i
      lid_to_part[i] = MPI.Comm_rank(comm.comm)+1
    end

    k=pXest.local_num_quadrants+1
    for i=1:pXest_ghost.mpisize
      for j=proc_offsets[i]:proc_offsets[i+1]-1
        quadrant       = ptr_ghost_quadrants[j+1]
        piggy3         = quadrant.p.piggy3
        lid_to_part[k] = i
        lid_to_gid[k]  = global_first_quadrant[i]+piggy3.local_num+1
        k=k+1
      end
    end
    # if (part==2)
    #     print(n,"\n")
    #     print(lid_to_gid,"\n")
    #     print(lid_to_part,"\n")
    # end
    IndexSet(n,lid_to_gid,lid_to_part)
  end
end

function setup_pXest_lnodes(::Type{Val{Dc}}, ptr_pXest, ptr_pXest_ghost) where Dc
  if (Dc==2)
    p4est_lnodes_new(ptr_pXest, ptr_pXest_ghost, Cint(1))
  else
    p8est_lnodes_new(ptr_pXest, ptr_pXest_ghost, Cint(1))
  end
end

function generate_cell_vertex_gids(ptr_pXest_lnodes, cellindices)
  pXest_lnodes=ptr_pXest_lnodes[]

  nvertices = pXest_lnodes.vnodes
  element_nodes = unsafe_wrap(Array,
                              pXest_lnodes.element_nodes,
                              pXest_lnodes.num_local_elements*nvertices)

  nonlocal_nodes = unsafe_wrap(Array,
                               pXest_lnodes.nonlocal_nodes,
                               pXest_lnodes.num_local_nodes-pXest_lnodes.owned_count)

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
    for i=1:pXest_lnodes.num_local_elements
      for j=1:nvertices
        l=element_nodes[k+j-1]
        if (l < pXest_lnodes.owned_count)
          data[current]=pXest_lnodes.global_offset+l+1
        else
          data[current]=nonlocal_nodes[l-pXest_lnodes.owned_count+1]+1
        end
        current=current+1
      end
      k=k+nvertices
    end
    Gridap.Arrays.Table(data,ptrs)
  end
  exchange!(cell_vertex_gids)
  cell_vertex_gids
end


function generate_cell_vertex_lids_nlvertices(cell_vertex_gids)
  DistributedData(cell_vertex_gids) do part, cell_vertex_gids
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
end

function generate_node_coordinates(::Type{Val{Dc}},
                                  cell_vertex_lids_nlvertices,
                                  ptr_pXest_connectivity,
                                  ptr_pXest,
                                  ptr_pXest_ghost) where Dc


  PXEST_CORNERS=2^Dc
  pXest_ghost = ptr_pXest_ghost[]
  pXest       = ptr_pXest[]

  # Obtain ghost quadrants
  if (Dc==2)
    ptr_ghost_quadrants = Ptr{p4est_quadrant_t}(pXest_ghost.ghosts.array)
  else
    ptr_ghost_quadrants = Ptr{p8est_quadrant_t}(pXest_ghost.ghosts.array)
  end

  tree_offsets = unsafe_wrap(Array, pXest_ghost.tree_offsets, pXest_ghost.num_trees+1)
  dnode_coordinates=DistributedData(cell_vertex_lids_nlvertices) do part, (cell_vertex_lids, nl)
     node_coordinates=Vector{Point{Dc,Float64}}(undef,nl)
     current=1
     vxy=Vector{Cdouble}(undef,Dc)
     pvxy=pointer(vxy,1)
     cell_lids=cell_vertex_lids.data
     for itree=1:pXest_ghost.num_trees
       if (Dc==2)
         tree = p4est_tree_array_index(pXest.trees, itree-1)[]
       else
         tree = p8est_tree_array_index(pXest.trees, itree-1)[]
       end
       for cell=1:tree.quadrants.elem_count
          if (Dc==2)
            quadrant=p4est_quadrant_array_index(tree.quadrants, cell-1)[]
          else
            quadrant=p8est_quadrant_array_index(tree.quadrants, cell-1)[]
          end
          for vertex=1:PXEST_CORNERS
             if (Dc==2)
               p4est_get_quadrant_vertex_coordinates(ptr_pXest_connectivity,
                                                     p4est_topidx_t(itree-1),
                                                     quadrant.x,
                                                     quadrant.y,
                                                     quadrant.level,
                                                     Cint(vertex-1),
                                                     pvxy)
             else
               p8est_get_quadrant_vertex_coordinates(ptr_pXest_connectivity,
                                                     p4est_topidx_t(itree-1),
                                                     quadrant.x,
                                                     quadrant.y,
                                                     quadrant.z,
                                                     quadrant.level,
                                                     Cint(vertex-1),
                                                     pvxy)
             end

            node_coordinates[cell_lids[current]]=Point{Dc,Float64}(vxy...)
            current=current+1
          end
       end
     end

     # Go over ghost cells
     for i=1:pXest_ghost.num_trees
      for j=tree_offsets[i]:tree_offsets[i+1]-1
          quadrant = ptr_ghost_quadrants[j+1]
          for vertex=1:PXEST_CORNERS
            if (Dc==2)
               p4est_get_quadrant_vertex_coordinates(ptr_pXest_connectivity,
                                                     p4est_topidx_t(i-1),
                                                     quadrant.x,
                                                     quadrant.y,
                                                     quadrant.level,
                                                     Cint(vertex-1),
                                                     pvxy)
            else
              p8est_get_quadrant_vertex_coordinates(ptr_pXest_connectivity,
                                                     p4est_topidx_t(i-1),
                                                     quadrant.x,
                                                     quadrant.y,
                                                     quadrant.z,
                                                     quadrant.level,
                                                     Cint(vertex-1),
                                                     pvxy)

            end
          #  if (MPI.Comm_rank(comm.comm)==0)
          #     println(vxy)
          #  end
           node_coordinates[cell_lids[current]]=Point{Dc,Float64}(vxy...)
           current=current+1
         end
       end
     end
     node_coordinates
  end
end

function generate_grid_and_topology(::Type{Val{Dc}},
                                    cell_vertex_lids_nlvertices,
                                    node_coordinates) where {Dc}
  DistributedData(cell_vertex_lids_nlvertices,node_coordinates) do part, (cell_vertex_lids, nl), node_coordinates
    polytope= Dc==2 ? QUAD : HEX
    scalar_reffe=Gridap.ReferenceFEs.ReferenceFE(polytope,Gridap.ReferenceFEs.lagrangian,Float64,1)
    cell_types=collect(Fill(1,length(cell_vertex_lids)))
    cell_reffes=[scalar_reffe]
    grid = Gridap.Geometry.UnstructuredGrid(node_coordinates,
                                            cell_vertex_lids,
                                            cell_reffes,
                                            cell_types,
                                            Gridap.Geometry.NonOriented())

    topology = Gridap.Geometry.UnstructuredGridTopology(node_coordinates,
                                      cell_vertex_lids,
                                      cell_types,
                                      map(Gridap.ReferenceFEs.get_polytope, cell_reffes),
                                      Gridap.Geometry.NonOriented())
    grid,topology
  end
end

function generate_face_labeling(comm,
                                cellindices,
                                coarse_discrete_model::DiscreteModel{Dc,Dp},
                                grid_and_topology,
                                ptr_pXest,
                                ptr_pXest_ghost) where {Dc,Dp}

  pXest       = ptr_pXest[]
  pXest_ghost = ptr_pXest_ghost[]

  coarse_grid_topology  = Gridap.Geometry.get_grid_topology(coarse_discrete_model)
  coarse_grid_labeling  = Gridap.Geometry.get_face_labeling(coarse_discrete_model)

  coarse_cell_vertices = Gridap.Geometry.get_faces(coarse_grid_topology,Dc,0)
  if (Dc==3)
    coarse_cell_edgets = Gridap.Geometry.get_faces(coarse_grid_topology,Dc,1)
  end
  coarse_cell_facets   = Gridap.Geometry.get_faces(coarse_grid_topology,Dc,Dc-1)

  owned_trees_offset=Vector{Int}(undef,pXest_ghost.num_trees+1)
  owned_trees_offset[1]=0
  for itree=1:pXest_ghost.num_trees
    if Dc==2
      tree = p4est_tree_array_index(pXest.trees, itree-1)[]
    else
      tree = p8est_tree_array_index(pXest.trees, itree-1)[]
    end
    owned_trees_offset[itree+1]=owned_trees_offset[itree]+tree.quadrants.elem_count
  end

  dfaces_to_entity=DistributedData(grid_and_topology) do part, (grid,topology)
     # Iterate over corners
     num_vertices=Gridap.Geometry.num_faces(topology,0)
     vertex_to_entity=zeros(Int,num_vertices)
     cell_vertices=Gridap.Geometry.get_faces(topology,Dc,0)
    #  if part==1
    #    println(cell_vertices)
    #  end

     # Corner iterator callback
     function jcorner_callback(pinfo     :: Ptr{p8est_iter_corner_info_t},
                               user_data :: Ptr{Cvoid})
        info=pinfo[]
        if (Dc==2)
          sides=Ptr{p4est_iter_corner_side_t}(info.sides.array)
        else
          sides=Ptr{p8est_iter_corner_side_t}(info.sides.array)
        end
        nsides=info.sides.elem_count
        tree=sides[1].treeid+1
        # We are on the interior of a tree
        data=sides[1]
        if data.is_ghost==1
           ref_cell=pXest.local_num_quadrants+data.quadid+1
        else
           ref_cell=owned_trees_offset[tree]+data.quadid+1
        end
        corner=sides[1].corner+1
        ref_cornergid=cell_vertices[ref_cell][corner]
        # if (MPI.Comm_rank(comm.comm)==0)
        #   println("XXX ", ref_cell, " ", ref_cornergid, " ", info.tree_boundary, " ", nsides, " ", corner)
        # end
        if (info.tree_boundary!=0 && nsides==1)
              # The current corner is also a corner of the coarse mesh
              coarse_cornergid=coarse_cell_vertices[tree][corner]
              vertex_to_entity[ref_cornergid]=
                 coarse_grid_labeling.d_to_dface_to_entity[1][coarse_cornergid]
        else
          if vertex_to_entity[ref_cornergid]==0
            # We are on the interior of a tree (if we did not touch it yet)
            vertex_to_entity[ref_cornergid]=coarse_grid_labeling.d_to_dface_to_entity[Dc+1][tree]
          end
        end
        # if (MPI.Comm_rank(comm.comm)==0)
        #   println("YYY ", cell_vertices)
        # end
        nothing
     end

     #  C-callable face callback
     ccorner_callback = @cfunction($jcorner_callback,
                                   Cvoid,
                                   (Ptr{p8est_iter_corner_info_t},Ptr{Cvoid}))

     cell_edgets=Gridap.Geometry.get_faces(topology,Dc,1)
     num_edgets=Gridap.Geometry.num_faces(topology,1)
     edget_to_entity=zeros(Int,num_edgets)
     if (Dc==3)
       # Edge iterator callback
       function jedge_callback(pinfo     :: Ptr{p8est_iter_edge_info_t},
                               user_data :: Ptr{Cvoid})
         info=pinfo[]
         sides=Ptr{p8est_iter_edge_side_t}(info.sides.array)

         nsides=info.sides.elem_count
         tree=sides[1].treeid+1
         # We are on the interior of a tree
         data=sides[1].is.full
         if data.is_ghost==1
           ref_cell=pXest.local_num_quadrants+data.quadid+1
         else
           ref_cell=owned_trees_offset[tree]+data.quadid+1
         end
         edge=sides[1].edge+1

         polytope=HEX
         poly_faces=Gridap.ReferenceFEs.get_faces(polytope)
         poly_edget_range=Gridap.ReferenceFEs.get_dimrange(polytope,1)
         poly_first_edget=first(poly_edget_range)
         poly_facet=poly_first_edget+edge-1

        #  if (MPI.Comm_rank(comm.comm)==0)
        #    coarse_edgetgid=coarse_cell_edgets[tree][edge]
        #    coarse_edgetgid_entity=coarse_grid_labeling.d_to_dface_to_entity[2][coarse_edgetgid]
        #    println("PPP ", ref_cell, " ", edge, " TB ", info.tree_boundary, " ", nsides, " ",coarse_edgetgid, " ",  coarse_edgetgid_entity)
        #  end
         if (info.tree_boundary!=0 && nsides==1)
          coarse_edgetgid=coarse_cell_edgets[tree][edge]
          coarse_edgetgid_entity=coarse_grid_labeling.d_to_dface_to_entity[2][coarse_edgetgid]
          # We are on the boundary of coarse mesh or inter-octree boundary
          for poly_incident_face in poly_faces[poly_facet]
            if poly_incident_face == poly_facet
              ref_edgetgid=cell_edgets[ref_cell][edge]
              edget_to_entity[ref_edgetgid]=coarse_edgetgid_entity
            else
              ref_cornergid=cell_vertices[ref_cell][poly_incident_face]
              # if (MPI.Comm_rank(comm.comm)==0)
              #    println("CCC ", ref_cell, " ", ref_cornergid, " ", info.tree_boundary, " ", nsides)
              # end
              vertex_to_entity[ref_cornergid]=coarse_edgetgid_entity
            end
          end
         else
          # We are on the interior of the domain if we did not touch the edge yet
          ref_edgetgid=cell_edgets[ref_cell][edge]
          if (edget_to_entity[ref_edgetgid]==0)
            edget_to_entity[ref_edgetgid]=coarse_grid_labeling.d_to_dface_to_entity[Dc+1][tree]
          end
         end
         nothing
       end

       # C-callable edge callback
       cedge_callback = @cfunction($jedge_callback,
                                   Cvoid,
                                   (Ptr{p8est_iter_edge_info_t},Ptr{Cvoid}))
     end

     # Iterate over faces
     num_faces=Gridap.Geometry.num_faces(topology,Dc-1)
     facet_to_entity=zeros(Int,num_faces)
     cell_facets=Gridap.Geometry.get_faces(topology,Dc,Dc-1)


     # Face iterator callback
     function jface_callback(pinfo     :: Ptr{p8est_iter_face_info_t},
                             user_data :: Ptr{Cvoid})
        info=pinfo[]
        if Dc==2
          sides=Ptr{p4est_iter_face_side_t}(info.sides.array)
        else
          sides=Ptr{p8est_iter_face_side_t}(info.sides.array)
        end

        nsides=info.sides.elem_count
        tree=sides[1].treeid+1
        # We are on the interior of a tree
        data=sides[1].is.full
        if data.is_ghost==1
           ref_cell=pXest.local_num_quadrants+data.quadid+1
        else
           ref_cell=owned_trees_offset[tree]+data.quadid+1
        end
        face=sides[1].face+1
        if Dc==2
          gridap_facet=P4EST_2_GRIDAP_FACET_2D[face]
        else
          gridap_facet=P4EST_2_GRIDAP_FACET_3D[face]
        end

        polytope= Dc==2 ? QUAD : HEX
        poly_faces=Gridap.ReferenceFEs.get_faces(polytope)
        poly_facet_range=Gridap.ReferenceFEs.get_dimrange(polytope,Dc-1)
        poly_first_facet=first(poly_facet_range)
        poly_facet=poly_first_facet+gridap_facet-1

        # if (MPI.Comm_rank(comm.comm)==0)
        #   coarse_facetgid=coarse_cell_facets[tree][gridap_facet]
        #   coarse_facetgid_entity=coarse_grid_labeling.d_to_dface_to_entity[Dc][coarse_facetgid]
        #   println("PPP ", ref_cell, " ", gridap_facet, " ", info.tree_boundary, " ", nsides, " ",coarse_facetgid, " ",  coarse_facetgid_entity)
        # end
        if (info.tree_boundary!=0 && nsides==1)
          coarse_facetgid=coarse_cell_facets[tree][gridap_facet]
          coarse_facetgid_entity=coarse_grid_labeling.d_to_dface_to_entity[Dc][coarse_facetgid]
          # We are on the boundary of coarse mesh or inter-octree boundary
          for poly_incident_face in poly_faces[poly_facet]
            if poly_incident_face == poly_facet
              ref_facetgid=cell_facets[ref_cell][gridap_facet]
              facet_to_entity[ref_facetgid]=coarse_facetgid_entity
            elseif (Dc==3 && poly_incident_face in Gridap.ReferenceFEs.get_dimrange(polytope,1))
              poly_first_edget=first(Gridap.ReferenceFEs.get_dimrange(polytope,1))
              edget=poly_incident_face-poly_first_edget+1
              ref_edgetgid=cell_edgets[ref_cell][edget]
              edget_to_entity[ref_edgetgid]=coarse_facetgid_entity
            else
              ref_cornergid=cell_vertices[ref_cell][poly_incident_face]
              # if (MPI.Comm_rank(comm.comm)==0)
              #    println("CCC ", ref_cell, " ", ref_cornergid, " ", info.tree_boundary, " ", nsides)
              # end
              vertex_to_entity[ref_cornergid]=coarse_facetgid_entity
            end
          end
        else
          # We are on the interior of the domain
          ref_facegid=cell_facets[ref_cell][gridap_facet]
          facet_to_entity[ref_facegid]=coarse_grid_labeling.d_to_dface_to_entity[Dc+1][tree]
        end
        nothing
     end

    #  C-callable face callback
    cface_callback = @cfunction($jface_callback,
                                 Cvoid,
                                 (Ptr{p8est_iter_face_info_t},Ptr{Cvoid}))


    # Iterate over cells
    num_cells=Gridap.Geometry.num_faces(topology,Dc)
    cell_to_entity=zeros(Int,num_cells)

    # Face iterator callback
    function jcell_callback(pinfo     :: Ptr{p8est_iter_volume_info_t},
                            user_data :: Ptr{Cvoid})
      info=pinfo[]
      tree=info.treeid+1
      cell=owned_trees_offset[tree]+info.quadid+1
      #println("XXX $(tree) $(cell)")
      cell_to_entity[cell]=coarse_grid_labeling.d_to_dface_to_entity[Dc+1][tree]
      nothing
    end
    ccell_callback = @cfunction($jcell_callback,
                                 Cvoid,
                                 (Ptr{p8est_iter_volume_info_t},Ptr{Cvoid}))

    if (Dc==2)
       p4est_iterate(ptr_pXest,ptr_pXest_ghost,C_NULL,C_NULL,cface_callback,C_NULL)
       p4est_iterate(ptr_pXest,ptr_pXest_ghost,C_NULL,ccell_callback,C_NULL,ccorner_callback)
    else
       p8est_iterate(ptr_pXest,ptr_pXest_ghost,C_NULL,C_NULL,cface_callback,C_NULL,C_NULL)
       p8est_iterate(ptr_pXest,ptr_pXest_ghost,C_NULL,C_NULL,C_NULL,cedge_callback,C_NULL)
       p8est_iterate(ptr_pXest,ptr_pXest_ghost,C_NULL,ccell_callback,C_NULL,C_NULL,ccorner_callback)
    end
    if (Dc==2)
      vertex_to_entity, facet_to_entity, cell_to_entity
    else
      vertex_to_entity, edget_to_entity, facet_to_entity, cell_to_entity
    end
 end

 dvertex_to_entity = DistributedData(comm,dfaces_to_entity) do part, faces
   faces[1]
 end
 if Dc==3
   dedget_to_entity = DistributedData(comm,dfaces_to_entity) do part, faces
    faces[2]
   end
 end
 dfacet_to_entity  = DistributedData(comm,dfaces_to_entity) do part, faces
   faces[Dc]
 end
 dcell_to_entity   = DistributedData(comm,dfaces_to_entity) do part, faces
   faces[Dc+1]
 end

 function dcell_to_faces(grid_and_topology,cell_dim,face_dim)
   DistributedData(get_comm(grid_and_topology),grid_and_topology) do part, (grid,topology)
    Gridap.Geometry.get_faces(topology,cell_dim,face_dim)
  end
 end

 polytope = Dc==2 ? QUAD : HEX

 update_face_to_entity_with_ghost_data!(dvertex_to_entity,
                                        cellindices,
                                        num_faces(polytope,0),
                                        dcell_to_faces(grid_and_topology,Dc,0))

 if Dc==3
   update_face_to_entity_with_ghost_data!(dedget_to_entity,
                                          cellindices,
                                          num_faces(polytope,1),
                                          dcell_to_faces(grid_and_topology,Dc,1))
 end

 update_face_to_entity_with_ghost_data!(dfacet_to_entity,
                                        cellindices,
                                        num_faces(polytope,Dc-1),
                                        dcell_to_faces(grid_and_topology,Dc,Dc-1))

 update_face_to_entity_with_ghost_data!(dcell_to_entity,
                                        cellindices,
                                        num_faces(polytope,Dc),
                                        dcell_to_faces(grid_and_topology,Dc,Dc))


 if (Dc==2)
  dfaces_to_entity=[dvertex_to_entity,dfacet_to_entity,dcell_to_entity]
 else
  dfaces_to_entity=[dvertex_to_entity,dedget_to_entity,dfacet_to_entity,dcell_to_entity]
 end


 dface_labeling =
  DistributedData(comm,dfaces_to_entity...) do part, faces_to_entity...

    # if (part == 1)
    #    println("XXX", faces_to_entity[1])
    #    println("XXX", faces_to_entity[2])
    #    println("XXX", faces_to_entity[3])
    #    #println("XXX", faces_to_entity[4])
    # end

    d_to_dface_to_entity       = Vector{Vector{Int}}(undef,Dc+1)
    d_to_dface_to_entity[1]    = faces_to_entity[1]
    if (Dc==3)
      d_to_dface_to_entity[2]  = faces_to_entity[2]
    end
    d_to_dface_to_entity[Dc]   = faces_to_entity[Dc]
    d_to_dface_to_entity[Dc+1] = faces_to_entity[Dc+1]
    Gridap.Geometry.FaceLabeling(d_to_dface_to_entity,
                                 coarse_grid_labeling.tag_to_entities,
                                 coarse_grid_labeling.tag_to_name)
 end
 dface_labeling
end



function p4est_connectivity_print(pconn::Ptr{p4est_connectivity_t})
  # struct p4est_connectivity
  #   num_vertices::p4est_topidx_t
  #   num_trees::p4est_topidx_t
  #   num_corners::p4est_topidx_t
  #   vertices::Ptr{Cdouble}
  #   tree_to_vertex::Ptr{p4est_topidx_t}
  #   tree_attr_bytes::Csize_t
  #   tree_to_attr::Cstring
  #   tree_to_tree::Ptr{p4est_topidx_t}
  #   tree_to_face::Ptr{Int8}
  #   tree_to_corner::Ptr{p4est_topidx_t}
  #   ctt_offset::Ptr{p4est_topidx_t}
  #   corner_to_tree::Ptr{p4est_topidx_t}
  #   corner_to_corner::Ptr{Int8}
  # end
  conn = pconn[]
  println("num_vertices=$(conn.num_vertices)")
  println("num_trees=$(conn.num_trees)")
  println("num_corners=$(conn.num_corners)")
  vertices=unsafe_wrap(Array, conn.vertices, conn.num_vertices*3)
  println("vertices=$(vertices)")
end

function init_cell_to_face_entity(part,
                                  num_faces_x_cell,
                                  cell_to_faces,
                                  face_to_entity)
  ptrs = Vector{eltype(num_faces_x_cell)}(undef,length(cell_to_faces) + 1)
  ptrs[2:end] .= num_faces_x_cell
  Gridap.Arrays.length_to_ptrs!(ptrs)
  data = Vector{eltype(face_to_entity)}(undef, ptrs[end] - 1)
  cell_to_face_entity = lazy_map(Broadcasting(Reindex(face_to_entity)),cell_to_faces)
  k = 1
  for i = 1:length(cell_to_face_entity)
    for j = 1:length(cell_to_face_entity[i])
      k=_fill_data!(data,cell_to_face_entity[i][j],k)
    end
  end
  return Gridap.Arrays.Table(data, ptrs)
end

function update_face_to_entity!(part,face_to_entity, cell_to_faces, cell_to_face_entity)
  for cell in 1:length(cell_to_faces)
      i_to_entity = cell_to_face_entity[cell]
      pini = cell_to_faces.ptrs[cell]
      pend = cell_to_faces.ptrs[cell + 1] - 1
      for (i, p) in enumerate(pini:pend)
        lid = cell_to_faces.data[p]
        face_to_entity[lid] = i_to_entity[i]
      end
  end
end

function update_face_to_entity_with_ghost_data!(
   face_to_entity,cell_gids,num_faces_x_cell,cell_to_faces)

   part_to_cell_to_entity = DistributedVector(init_cell_to_face_entity,
                                               cell_gids,
                                               num_faces_x_cell,
                                               cell_to_faces,
                                               face_to_entity)
    exchange!(part_to_cell_to_entity)
    do_on_parts(update_face_to_entity!,
                get_comm(cell_gids),
                face_to_entity,
                cell_to_faces,
                part_to_cell_to_entity)
end


function UniformlyRefinedForestOfOctreesDiscreteModel(comm::Communicator,
                                                      coarse_discrete_model::DiscreteModel{Dc,Dp},
                                                      num_uniform_refinements::Int) where {Dc,Dp}

  ptr_pXest_connectivity=setup_pXest_connectivity(coarse_discrete_model)

  # Create a new forest
  ptr_pXest = setup_pXest(Val{Dc},comm,ptr_pXest_connectivity,num_uniform_refinements)

  # Build the ghost layer
  ptr_pXest_ghost=setup_pXest_ghost(Val{Dc},ptr_pXest)

  cellindices = setup_cell_indexset(Val{Dc},comm,ptr_pXest,ptr_pXest_ghost)

  ptr_pXest_lnodes=setup_pXest_lnodes(Val{Dc}, ptr_pXest, ptr_pXest_ghost)

  cell_vertex_gids=generate_cell_vertex_gids(ptr_pXest_lnodes,cellindices)

  cell_vertex_lids_nlvertices=generate_cell_vertex_lids_nlvertices(cell_vertex_gids)

  dnode_coordinates=generate_node_coordinates(Val{Dc},
                                              cell_vertex_lids_nlvertices,
                                              ptr_pXest_connectivity,
                                              ptr_pXest,
                                              ptr_pXest_ghost)

  dgrid_and_topology=generate_grid_and_topology(Val{Dc},
                       cell_vertex_lids_nlvertices,dnode_coordinates)


  dface_labeling=generate_face_labeling(comm,
                       cellindices,
                       coarse_discrete_model,
                       dgrid_and_topology,
                       ptr_pXest,
                       ptr_pXest_ghost)

  ddiscretemodel=
    DistributedData(comm,dgrid_and_topology,dface_labeling) do part, (grid,topology), face_labeling
      Gridap.Geometry.UnstructuredDiscreteModel(grid,topology,face_labeling)
    end

  # Write forest to VTK file
  #p4est_vtk_write_file(unitsquare_forest, C_NULL, "my_step")

  if (Dc==2)
    # Destroy lnodes
    p4est_lnodes_destroy(ptr_pXest_lnodes)
    # Destroy ghost
    p4est_ghost_destroy(ptr_pXest_ghost)
    # Destroy the forest
    p4est_destroy(ptr_pXest)
    # Destroy the connectivity
    p4est_connectivity_destroy(ptr_pXest_connectivity)
  else
    # Destroy lnodes
    p8est_lnodes_destroy(ptr_pXest_lnodes)
    # Destroy ghost
    p8est_ghost_destroy(ptr_pXest_ghost)
    # Destroy the forest
    p8est_destroy(ptr_pXest)
    # Destroy the connectivity
    p8est_connectivity_destroy(ptr_pXest_connectivity)
  end

  DistributedDiscreteModel(ddiscretemodel,cellindices)

end
