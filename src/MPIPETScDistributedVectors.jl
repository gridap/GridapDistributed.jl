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

function DistributedVector(
  initializer::Function, indices::MPIPETScDistributedIndexSet, args...)
  comm = get_comm(indices)
  data = DistributedData(initializer, comm, args...)
  part = data.part
  if (eltype(part) <: Number)
    indices,vecghost = _create_eltype_number_indices_ghost(part,indices)
  else
    @assert eltype(part) <: AbstractVector{<:Number}
    indices,vecghost = _create_eltype_vector_number_indices_ghost(part,indices)
  end
  MPIPETScDistributedVector(part,indices,vecghost)
end

function DistributedVector{T}(
  initializer::Function, indices::MPIPETScDistributedIndexSet, args...) where T <: Union{Number,AbstractVector{<:Number}}
  comm = get_comm(indices)
  data = DistributedData(initializer, comm, args...)
  part = data.part
  if (T <: Number)
    indices,vecghost = _create_eltype_number_indices_ghost(part,indices)
  else
    indices,vecghost = _create_eltype_vector_number_indices_ghost(part,indices)
  end
  MPIPETScDistributedVector(part,indices,vecghost)
end


function _create_eltype_number_indices_ghost(
  part::Vector{T},
  indices::MPIPETScDistributedIndexSet,
) where {T<:Number}
  @assert sizeof(eltype(part)) == sizeof(Float64)
  @assert length(part) == length(indices.parts.part.lid_to_owner)
  vecghost = create_ghost_vector(indices)
  indices, vecghost
end

function _create_eltype_vector_number_indices_ghost(
  local_part::Vector{T},
  indices::MPIPETScDistributedIndexSet,
) where {T<:AbstractVector{<:Number}}

@assert sizeof(eltype(T)) == sizeof(Float64)

l = length(local_part[1])
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

function unpack_all_entries!(a::MPIPETScDistributedVector{T}) where T
  num_local = length(a.indices.app_to_petsc_locidx)
  lvec = PETSc.LocalVector(a.vecghost,num_local)
  _unpack_all_entries!(T,a.part,a.indices.app_to_petsc_locidx,lvec)
  PETSc.restore(lvec)
end

function Base.getindex(a::MPIPETScDistributedVector,indices::MPIPETScDistributedIndexSet)
  @notimplementedif a.indices !== indices
  exchange!(a.vecghost)
  unpack_all_entries!(a)
  a
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
  _pack_local_entries!(a.vecghost, local_part, lid_to_owner, comm_rank)

  exchange!(a.vecghost)

  # Unpack data
  num_local = length(lid_to_owner)
  lvec = PETSc.LocalVector(a.vecghost,num_local)
  _unpack_ghost_entries!(eltype(local_part), local_part, lid_to_owner, comm_rank, app_to_petsc_locidx, lvec)
  PETSc.restore(lvec)
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
