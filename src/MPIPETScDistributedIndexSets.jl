# Specializations
struct MPIPETScDistributedIndexSet{A,B,C} <: DistributedIndexSet
  ngids               :: Int
  parts               :: MPIPETScDistributedData{IndexSet{A,B,C}}
  lid_to_gid_petsc    :: Vector{Int64}
  petsc_to_app_locidx :: Vector{Int32}
  app_to_petsc_locidx :: Vector{Int32}
end

num_gids(a::MPIPETScDistributedIndexSet) = a.ngids

get_part(
  comm::MPIPETScCommunicator,
  a::MPIPETScDistributedIndexSet,
  part::Integer) = a.parts.part

get_comm(a::MPIPETScDistributedIndexSet) = a.parts.comm

function DistributedIndexSet(initializer::Function,comm::MPIPETScCommunicator,ngids::Integer,args...)
  parts = DistributedData(initializer,comm,args...)
  lid_to_gid_petsc, petsc_to_app_locidx, app_to_petsc_locidx = _compute_internal_members(comm,parts.part)
  MPIPETScDistributedIndexSet(ngids,parts,lid_to_gid_petsc,petsc_to_app_locidx,app_to_petsc_locidx)
end

#TODO: think about type stability of this auxiliary function
function _compute_internal_members(comm::MPIPETScCommunicator, is::IndexSet)
  comm_rank = MPI.Comm_rank(comm.comm)
  num_owned_entries = count( (a)->(a==comm_rank+1), is.lid_to_owner )
  num_local_entries = length(is.lid_to_owner)

  sndbuf = Ref{Int64}(num_owned_entries)
  rcvbuf = Ref{Int64}()
  MPI.Exscan!(sndbuf, rcvbuf, 1, +, comm.comm)


  app_idx = Array{Int64}(undef, num_owned_entries)
  petsc_idx = Array{Int64}(undef, num_owned_entries)

  (comm_rank == 0) ? (offset = 1) : (offset = rcvbuf[] + 1)
  current = 1
  for i = 1:num_local_entries
    if (is.lid_to_owner[i] == comm_rank + 1)
      app_idx[current] = is.lid_to_gid[i]
      petsc_idx[current] = offset
      offset = offset + 1
      current = current + 1
    end
  end

  println(app_idx, petsc_idx)


  # build application ordering in order to get lid_to_gid
  # accordingly to PETSc global numbering
  petsc_ao = AO(Float64, app_idx, petsc_idx)
  lid_to_gid_petsc = collect(is.lid_to_gid)
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


# for i=1:num_local_entries
#   if (lid_to_owner[i]!==comm_rank+1)
#      push!(ghost_idx, lid_to_gid_petsc[i])
#   end
# end
#
# vec = VecGhost(Float64, num_owned_entries, ghost_idx)
# lvec = LocalVector(vec)
# for i=1:num_owned_entries
#   lvec.a[i]=reinterpret(Float64,lid_to_gid[petsc_to_app_locidx[i]])
# end
#
# restore(lvec)
# scatter!(vec)
#
# lvecghost = VecLocal(vec)
# lvec      = LocalVector(lvecghost)
# println("$(comm_rank): $(reinterpret(Int64,lvec.a))")
# restore(lvec)
# restore(lvecghost)
