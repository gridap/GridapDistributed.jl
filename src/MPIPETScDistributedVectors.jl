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

function Base.getindex(a::MPIPETScDistributedVector,indices::MPIPETScDistributedIndexSet)
  @notimplementedif a.indices !== indices
  exchange!(a)
  DistributedVector(indices,indices,a) do part, indices, a
      a[indices.lid_to_gid]
  end
end

function exchange!(a::MPIPETScDistributedVector{T}) where T
  indices = a.indices
  local_part = a.part
  lid_to_owner = indices.parts.part.lid_to_owner
  petsc_to_app_locidx = indices.petsc_to_app_locidx
  app_to_petsc_locidx = indices.app_to_petsc_locidx
  comm = get_comm(indices)
  comm_rank = MPI.Comm_rank(comm.comm)

  num_owned = num_owned_entries(a.indices)
  num_local = length(lid_to_owner)
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
  set_values_local!(a.vecghost, idxs, vals, PETSc.C.INSERT_VALUES)

  AssemblyBegin(a.vecghost, PETSc.C.MAT_FINAL_ASSEMBLY)
  AssemblyEnd(a.vecghost, PETSc.C.MAT_FINAL_ASSEMBLY)

  # Send data
  PETSc.scatter!(a.vecghost)

  # Unpack data
  lvec = PETSc.LocalVector(a.vecghost,num_local)
  _unpack!(eltype(local_part), local_part, lid_to_owner, comm_rank, app_to_petsc_locidx, lvec)
  PETSc.restore(lvec)
end

function _unpack!(
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

function _unpack!(
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
