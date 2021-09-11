using MPI
using p4est_wrapper
using Gridap
using GridapDistributed
using Test
using FillArrays

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
    if corner == 0
       #neighbour[] = myself[]
    elseif corner == 1
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


MPIPETScCommunicator() do comm
  @test num_parts(comm) == 2

  # Initialize MPI if not initialized yet
  if !MPI.Initialized()
     MPI.Init()
  end

  #############################################################################
  # Main program
  #############################################################################

  mpicomm = comm.comm #p4est_wrapper.P4EST_ENABLE_MPI ? MPI.COMM_WORLD : Cint(0)

  # Create a connectivity structure for the unit square.
  unitsquare_connectivity = p4est_connectivity_new_unitsquare()
  @test unitsquare_connectivity != C_NULL

  # Create a new forest
  unitsquare_forest = p4est_new_ext(mpicomm,
                                    unitsquare_connectivity,
                                    Cint(0), Cint(2), Cint(1), Cint(0),
                                    C_NULL, C_NULL)
  @test unitsquare_forest != C_NULL

  unitsquare_ghost=p4est_ghost_new(unitsquare_forest,p4est_wrapper.P4EST_CONNECT_FULL)
  @test unitsquare_ghost != C_NULL

  # Build the ghost layer.
  p4est_ghost = unitsquare_ghost[]
  p4est       = unitsquare_forest[]

  # Obtain ghost quadrants
  ##p4est_quadrant_t * ptr_ghost_quadrants = (p4est_quadrant_t *) p4est_ghost->ghosts.array;
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
  @test ptr_unitsquare_lnodes != C_NULL

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
             if (MPI.Comm_rank(comm.comm)==0)
                println(vxy)
             end
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

  dgrid=DistributedData(cell_vertex_lids_nlvertices,dnode_coordinates) do part, (cell_vertex_lids, nl), node_coordinates
      scalar_reffe=Gridap.ReferenceFEs.ReferenceFE(QUAD,Gridap.ReferenceFEs.lagrangian,Float64,1)
      cell_types=collect(Fill(1,length(cell_vertex_lids)))
      cell_reffes=[scalar_reffe]
      grid = Gridap.Geometry.UnstructuredGrid(node_coordinates,
                                              cell_vertex_lids,
                                              cell_reffes,
                                              cell_types,
                                              Gridap.Geometry.Oriented())

      writevtk(grid,"grid$(part)")
      grid
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

  # Finalize MPI if initialized and session is not interactive
  # if (MPI.Initialized() && !isinteractive())
  #   MPI.Finalize()
  # end
end
