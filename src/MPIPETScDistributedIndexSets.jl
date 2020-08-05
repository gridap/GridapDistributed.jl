# Specializations
struct MPIPETScDistributedIndexSet{A,B,C} <: DistributedIndexSet
  ngids               :: Int
  # TO-THINK: Do we really need to have a DistributedData for parts?
  #           Why dont we store part directly?
  parts               :: MPIPETScDistributedData{IndexSet{A,B,C}}
  # TO-THINK: Should we store these as DistributedData?
  lid_to_gid_petsc    :: Vector{PETSc.C.PetscInt}
  IS                  :: PETSc.ISLocalToGlobalMapping{Float64}
  petsc_to_app_locidx :: Vector{Int32}
  app_to_petsc_locidx :: Vector{Int32}
end

num_gids(a::MPIPETScDistributedIndexSet) = a.ngids

get_part(
  comm::MPIPETScCommunicator,
  a::MPIPETScDistributedIndexSet,
  part::Integer) = a.parts.part

function num_owned_entries(indices::MPIPETScDistributedIndexSet)
    comm = get_comm(indices)
    comm_rank = MPI.Comm_rank(comm.comm) + 1
    lid_to_owner = indices.parts.part.lid_to_owner
    count((a)->(a == comm_rank), lid_to_owner)
  end

function create_ghost_vector(entries::Vector{Float64},
                             indices::MPIPETScDistributedIndexSet)
  ghost_idx = _generate_ghost_idx(indices)
  num_owned = num_owned_entries(indices)
  comm = get_comm(indices)
  vref = Ref{PETSc.C.Vec{Float64}}()
  PETSc.C.chk(PETSc.C.VecCreateGhostWithArray(comm.comm,
                          PETSc.C.PetscInt(num_owned),
                          PETSc.C.PetscInt(PETSc.C.PETSC_DECIDE),
                          PETSc.C.PetscInt(length(ghost_idx)),
                          ghost_idx,
                          entries,
                          vref))
  PETSc.C.chk(PETSc.C.VecSetType(vref[], PETSc.C.VECMPI))
  return PETSc.Vec{Float64}(vref[],entries)
end

function _generate_ghost_idx(indices::MPIPETScDistributedIndexSet)
  comm = get_comm(indices)
  comm_rank = MPI.Comm_rank(comm.comm)
  ghost_idx = PETSc.C.PetscInt[]
  lid_to_owner = indices.parts.part.lid_to_owner
  lid_to_gid_petsc = indices.lid_to_gid_petsc
  num_local_entries = length(lid_to_owner)
  for i = 1:num_local_entries
    if (lid_to_owner[i] !== comm_rank + 1)
      push!(ghost_idx, lid_to_gid_petsc[i]-1)
    end
  end
  ghost_idx
end



get_comm(a::MPIPETScDistributedIndexSet) = a.parts.comm

function DistributedIndexSet(initializer::Function,comm::MPIPETScCommunicator,ngids::Integer,args...)
  parts = DistributedData(initializer,comm,args...)
  lid_to_gid_petsc, petsc_to_app_locidx, app_to_petsc_locidx = _compute_internal_members(comm,parts.part)
  IS=PETSc.ISLocalToGlobalMapping(Float64, lid_to_gid_petsc, comm=comm.comm)
  MPIPETScDistributedIndexSet(ngids,parts,lid_to_gid_petsc, IS,petsc_to_app_locidx,app_to_petsc_locidx)
end

#TODO: think about type stability of this auxiliary function
function _compute_internal_members(comm::MPIPETScCommunicator, is::IndexSet)
  comm_rank = MPI.Comm_rank(comm.comm)
  num_owned_entries = count( (a)->(a==comm_rank+1), is.lid_to_owner )
  num_local_entries = length(is.lid_to_owner)

  sndbuf = Ref{Int64}(num_owned_entries)
  rcvbuf = Ref{Int64}()
  MPI.Exscan!(sndbuf, rcvbuf, 1, +, comm.comm)


  app_idx = Array{PETSc.C.PetscInt}(undef, num_owned_entries)
  petsc_idx = Array{PETSc.C.PetscInt}(undef, num_owned_entries)

  (comm_rank == 0) ? (offset = PETSc.C.PetscInt(1)) : (offset = PETSc.C.PetscInt(rcvbuf[] + 1))
  current = 1
  for i = 1:num_local_entries
    if (is.lid_to_owner[i] == comm_rank + 1)
      app_idx[current] = is.lid_to_gid[i]
      petsc_idx[current] = offset
      offset = offset + 1
      current = current + 1
    end
  end

  # build application ordering in order to get lid_to_gid
  # accordingly to PETSc global numbering
  petsc_ao = AO(Float64, app_idx, petsc_idx)
  lid_to_gid_petsc = convert(Vector{PETSc.C.PetscInt}, collect(is.lid_to_gid))
  map_app_to_petsc!(petsc_ao, lid_to_gid_petsc)

  ghost_idx = Int[]

  app_to_petsc_locidx = Vector{Int32}(undef, num_local_entries)
  current = 1
  for i = 1:num_local_entries
    if (is.lid_to_owner[i] == (comm_rank + 1))
      app_to_petsc_locidx[i] = current
      current = current + 1
    end
  end
  for i = 1:num_local_entries
    if (is.lid_to_owner[i] !== (comm_rank + 1))
      app_to_petsc_locidx[i] = current
      current = current + 1
    end
  end

  petsc_to_app_locidx = Vector{Int32}(undef, num_local_entries)
  for i = 1:num_local_entries
    petsc_to_app_locidx[app_to_petsc_locidx[i]] = i
  end
  return lid_to_gid_petsc, petsc_to_app_locidx, app_to_petsc_locidx
end
