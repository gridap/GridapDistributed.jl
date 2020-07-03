# Specializations
struct MPIPETScDistributedVector{T<:Union{Number,AbstractVector{<:Number}},V<:AbstractVector{T},A,B,C} <: DistributedVector{T}
  part :: V
  indices :: MPIPETScDistributedIndexSet{A,B,C}
  vecghost :: PETSc.Vec{Float64}
end

function MPIPETScDistributedVector{T}(
  part::V,
  indices::MPIPETScDistributedIndexSet,
  vecghost::PETSc.Vec{Float64}) where {T<:Number, V<:AbstractVector{T}}
  MPIPETScDistributedVector(part,indices,vecghost)
end

get_comm(a::MPIPETScDistributedVector) = get_comm(a.indices)

get_part(
  comm::MPIPETScCommunicator,
  a::MPIPETScDistributedVector,
  part::Integer) = a.part

get_part(
  comm::MPIPETScCommunicator,
  a::PETSc.Vec{Float64},
  part::Integer) = a

function DistributedVector{T}(
  indices::MPIPETScDistributedIndexSet) where T <: Number
  indices,vecghost = _create_eltype_number_indices_ghost(T,indices)
  lvecghost = PETSc.LocalVector(vecghost, length(indices.parts.part.lid_to_owner))
  a_reint=reinterpret(T,lvecghost.a)
  part=reindex(a_reint,indices.app_to_petsc_locidx)
  PETSc.restore(lvecghost)
  MPIPETScDistributedVector(part,indices,vecghost)
end

function _build_local_part_from_ptrs(::Type{T}, indices, block_indices, vecghost, ptrs) where T
  lvecghost = PETSc.LocalVector(vecghost, length(indices.parts.part.lid_to_owner))
  a_reint=reinterpret(eltype(T), lvecghost.a)
  TSUB=SubArray{eltype(T),1,typeof(a_reint),Tuple{UnitRange{Int64}},true}
  part=TSUB[ view(a_reint,ptrs[i]:ptrs[i+1]-1) for i=1:length(ptrs)-1 ]
  part=reindex(part,block_indices.app_to_petsc_locidx)
  PETSc.restore(lvecghost)
  part
end

function DistributedVector{T}(
  initializer::Function, indices::MPIPETScDistributedIndexSet,args...) where T
  @notimplemented "Unsupported method"
end
  
function DistributedVector(
  initializer::Function, indices::MPIPETScDistributedIndexSet,args...)
  @notimplemented "Unsupported method"
end


function DistributedVector{T}(
  indices::MPIPETScDistributedIndexSet, length_entry :: Int ) where T <: AbstractVector{<:Number}
  num_entries = length(indices.parts.part.lid_to_owner)
  ptrs=Vector{Int32}(undef,num_entries+1)
  ptrs[1]=1
  for i=1:num_entries
    ptrs[i+1]=ptrs[i]+length_entry
  end
  block_indices = indices
  indices,vecghost = _create_eltype_vector_number_indices_ghost(T,length_entry,indices)
  part = _build_local_part_from_ptrs(T, indices, block_indices, vecghost, ptrs)
  MPIPETScDistributedVector(part,indices,vecghost)
end

function DistributedVector{T}(
  indices::MPIPETScDistributedIndexSet, length_entries :: MPIPETScDistributedData ) where T <: AbstractVector{<:Number}
  num_entries   = length(indices.parts.part.lid_to_owner)
  ptrs=Vector{Int32}(undef,num_entries+1)
  @assert num_entries == length(length_entries.part)
  ptrs[1]=1
  for i=1:num_entries
    ptrs[i+1]=ptrs[i]+length_entries.part[i]
  end
  block_indices = indices
  indices,vecghost = _create_eltype_vector_number_variable_length_indices_ghost(T,length_entries.part,indices)
  part = _build_local_part_from_ptrs(T, indices, block_indices, vecghost, ptrs)
  MPIPETScDistributedVector(part,indices,vecghost)
end


function _create_eltype_number_indices_ghost(
  eltype::Type{T},
  indices::MPIPETScDistributedIndexSet,
) where {T<:Number}
  @assert sizeof(T) == sizeof(Float64)
  vecghost = create_ghost_vector(indices)
  indices, vecghost
end

function _create_eltype_vector_number_indices_ghost(
  eltype::Type{T},
  length_entry::Int,
  indices::MPIPETScDistributedIndexSet,
) where T<:AbstractVector{<:Number}

#println(T)
#println(eltype(T))
#@assert sizeof(eltype(T)) == sizeof(Float64)

l = length_entry
n = l * indices.ngids

indices = DistributedIndexSet(get_comm(indices),n,indices,l,n) do part, indices, l, n
  lid_to_gid   = Vector{Int}(undef, l*length(indices.lid_to_owner))
  lid_to_owner = Vector{Int}(undef, l*length(indices.lid_to_owner))
  k=1
  for i=1:length(indices.lid_to_owner)
     offset = (indices.lid_to_gid[i]-1)*l
     for j=1:l
        lid_to_gid[k]   = offset + j
        lid_to_owner[k] = indices.lid_to_owner[i]
        k=k+1
     end
  end
  IndexSet(n, lid_to_gid, lid_to_owner)
end
vecghost = create_ghost_vector(indices)
indices, vecghost
end

function _create_eltype_vector_number_variable_length_indices_ghost(
  eltype::Type{T},
  length_entries::AbstractVector{<:Integer},
  indices::MPIPETScDistributedIndexSet,
) where T<:AbstractVector{<:Number}

#println(T)
#println(eltype(T))
#@assert sizeof(eltype(T)) == sizeof(Float64)
comm      = get_comm(indices)
part      = MPI.Comm_rank(comm.comm)+1
num_owned_entries = 0
num_local_entries = 0
lid_to_owner = indices.parts.part.lid_to_owner
for i = 1:length(lid_to_owner)
   if (lid_to_owner[i] == part)
     num_owned_entries = num_owned_entries + length_entries[i]
   end
   num_local_entries = num_local_entries + length_entries[i]
end

sndbuf = Ref{Int64}(num_owned_entries)
rcvbuf = Ref{Int64}()

MPI.Exscan!(sndbuf, rcvbuf, 1, +, comm.comm)
(part == 1) ? (offset = 1) : (offset = rcvbuf[] + 1)

sendrecvbuf = Ref{Int64}(num_owned_entries)
MPI.Allreduce!(sendrecvbuf, +, comm.comm)
n = sendrecvbuf[]

offsets = DistributedVector{Int}(indices)
do_on_parts(offsets, indices) do part, offsets, indices
  for i=1:length(indices.lid_to_owner)
     if (indices.lid_to_owner[i] == part)
       offsets[i] = offset
       for j=1:length_entries[i]
         offset=offset+1
       end
     end
  end
end
exchange!(offsets)

indices = DistributedIndexSet(get_comm(indices),n,indices,offsets) do part, indices, offsets
  lid_to_gid   = Vector{Int}(undef, num_local_entries)
  lid_to_owner = Vector{Int}(undef, num_local_entries)
  k=1
  for i=1:length(indices.lid_to_owner)
     offset=offsets[i]
     for j=1:length_entries[i]
        lid_to_gid[k]   = offset
        lid_to_owner[k] = indices.lid_to_owner[i]
        k=k+1
        offset=offset+1
     end
  end
  IndexSet(n, lid_to_gid, lid_to_owner)
end
vecghost = create_ghost_vector(indices)
indices, vecghost
end

function unpack_all_entries!(a::MPIPETScDistributedVector{T}) where T
  num_local = length(a.indices.app_to_petsc_locidx)
  lvec = PETSc.LocalVector(a.vecghost,num_local)
  _unpack_all_entries!(T,a.part,a.indices.app_to_petsc_locidx,lvec)
  PETSc.restore(lvec)
end

function Base.getindex(a::PETSc.Vec{Float64},indices::MPIPETScDistributedIndexSet)
  result= DistributedVector{Float64}(indices)
  copy!(result.vecghost,a)
  exchange!(result.vecghost)
  result
end

function exchange!(a::PETSc.Vec{T}) where T
  # Send data
  PETSc.scatter!(a)
end

function exchange!(a::MPIPETScDistributedVector{T}) where T
  indices = a.indices
  local_part = a.part
  lid_to_owner = indices.parts.part.lid_to_owner
  petsc_to_app_locidx = indices.petsc_to_app_locidx
  app_to_petsc_locidx = indices.app_to_petsc_locidx
  comm = get_comm(indices)
  comm_rank = MPI.Comm_rank(comm.comm)

  # Pack data
  #_pack_local_entries!(a.vecghost, local_part, lid_to_owner, comm_rank)

  exchange!(a.vecghost)

  #  Unpack data
  # num_local = length(lid_to_owner)
  # lvec = PETSc.LocalVector(a.vecghost,num_local)
  # _unpack_ghost_entries!(eltype(local_part), local_part, lid_to_owner, comm_rank, app_to_petsc_locidx, lvec)
  # PETSc.restore(lvec)
end

function _pack_local_entries!(vecghost, local_part, lid_to_owner, comm_rank)
  num_owned = PETSc.lengthlocal(vecghost)
  idxs = Vector{Int}(undef, num_owned)
  vals = Vector{Float64}(undef, num_owned)
  # Pack data
  k=1
  current=1
  for i=1:length(local_part)
    for j=1:length(local_part[i])
       if ( lid_to_owner[k] == comm_rank+1 )
           idxs[current]=current-1
           vals[current]=reinterpret(Float64,local_part[i][j])
           current = current + 1
       end
       k=k+1
    end
  end
  set_values_local!(vecghost, idxs, vals, PETSc.C.INSERT_VALUES)
  AssemblyBegin(vecghost, PETSc.C.MAT_FINAL_ASSEMBLY)
  AssemblyEnd(vecghost, PETSc.C.MAT_FINAL_ASSEMBLY)
end

function _pack_all_entries!(vecghost, local_part)
  num_local = prod(size(local_part))
  idxs = Vector{Int}(undef, num_local)
  vals = Vector{Float64}(undef, num_local)

  # Pack data
  k = 1
  current = 1
  for i = 1:length(local_part)
    for j = 1:length(local_part[i])
      idxs[current] = current - 1
      vals[current] = reinterpret(Float64, local_part[i][j])
      current = current + 1
      k = k + 1
    end
  end
  set_values_local!(vecghost, idxs, vals, PETSc.C.INSERT_VALUES)
  AssemblyBegin(vecghost, PETSc.C.MAT_FINAL_ASSEMBLY)
  AssemblyEnd(vecghost, PETSc.C.MAT_FINAL_ASSEMBLY)
end

function _unpack_ghost_entries!(
  T::Type{<:Number},
  local_part,
  lid_to_owner,
  comm_rank,
  app_to_petsc_locidx,
  lvec,
)
  for i = 1:length(local_part)
    if (lid_to_owner[i] != comm_rank + 1)
      local_part[i] = reinterpret(T, lvec.a[app_to_petsc_locidx[i]])
    end
  end
end

function _unpack_ghost_entries!(
  T::Type{<:AbstractVector{K}},
  local_part,
  lid_to_owner,
  comm_rank,
  app_to_petsc_locidx,
  lvec,
) where K <: Number
  k = 1
  for i = 1:length(local_part)
    for j = 1:length(local_part[i])
      if (lid_to_owner[k] != comm_rank + 1)
        local_part[i][j] = reinterpret(K, lvec.a[app_to_petsc_locidx[k]])
      end
      k = k + 1
    end
  end
end

function _unpack_all_entries!(
  T::Type{<:Number},
  local_part,
  app_to_petsc_locidx,
  lvec,
)
  for i = 1:length(local_part)
    local_part[i] = reinterpret(T, lvec.a[app_to_petsc_locidx[i]])
  end
end

function _unpack_all_entries!(
  T::Type{<:AbstractVector{K}},
  local_part,
  app_to_petsc_locidx,
  lvec,
) where {K<:Number}
  k = 1
  for i = 1:length(local_part)
    for j = 1:length(local_part[i])
      local_part[i][j] = reinterpret(K, lvec.a[app_to_petsc_locidx[k]])
      k = k + 1
    end
  end
end

function Base.setindex!(a::Gridap.Arrays.Reindexed,v,j::Integer)
  i = a.j_to_i[j]
  a.i_to_v[i]=v
end
