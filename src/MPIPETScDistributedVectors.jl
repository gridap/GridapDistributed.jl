# Specializations
struct MPIPETScDistributedVector{T<:Union{Number,AbstractVector{K} where K<:Number},V<:AbstractVector{T},A,B,C}
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


function _create_eltype_number_indices_ghost(
  part::Vector{T},
  indices::MPIPETScDistributedIndexSet,
) where {T<:Number}
  @assert sizeof(eltype(part)) == sizeof(Float64)
  @assert length(part) == length(indices.parts.part.lid_to_owner)
  vecghost = _create_ghost_vector(indices)
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
vecghost = _create_ghost_vector(indices)
indices, vecghost
end


function _create_ghost_vector(indices::MPIPETScDistributedIndexSet)
  comm = get_comm(indices)
  comm_rank = MPI.Comm_rank(comm.comm)
  ghost_idx=Int[]
  lid_to_owner = indices.parts.part.lid_to_owner
  lid_to_gid_petsc  = indices.lid_to_gid_petsc
  num_local_entries = length(lid_to_owner)
  for i=1:num_local_entries
    if (lid_to_owner[i]!==comm_rank+1)
       push!(ghost_idx, lid_to_gid_petsc[i])
    end
  end
  num_owned_entries = _num_owned_entries(indices)
  VecGhost(Float64, num_owned_entries, ghost_idx; comm=comm.comm, vtype=PETSc.C.VECMPI)
end

function _num_owned_entries(indices::MPIPETScDistributedIndexSet)
  comm = get_comm(indices)
  comm_rank = MPI.Comm_rank(comm.comm)+1
  lid_to_owner = indices.parts.part.lid_to_owner
  count( (a)->(a==comm_rank), lid_to_owner )
end

function Base.getindex(a::MPIPETScDistributedVector,indices::MPIPETScDistributedIndexSet)
  @notimplementedif a.indices !== indices
  exchange!(a)
  a
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
  lvecghost = PETSc.VecLocal(a.vecghost)
  lvec      = PETSc.LocalVector(lvecghost)
  k=1
  current=1
  for i=1:length(local_part)
    for j=1:length(local_part[i])
       if ( lid_to_owner[k] == comm_rank+1 )
           lvec.a[current]=reinterpret(Float64,local_part[i][j])
           current = current + 1
       end
       k=k+1
    end
  end
  PETSc.restore(lvec)
  PETSc.restore(lvecghost)

  # Send data
  PETSc.scatter!(a.vecghost)

  # Unpack data
  lvecghost = PETSc.VecLocal(a.vecghost)
  lvec      = PETSc.LocalVector(lvecghost)
  _unpack!(eltype(local_part), local_part, lid_to_owner, comm_rank, app_to_petsc_locidx, lvec)
  PETSc.restore(lvec)
  PETSc.restore(lvecghost)
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

# Assembly related
# function Gridap.Algebra.allocate_vector(::Type{V},gids::DistributedIndexSet) where V <: AbstractVector
#   ngids = num_gids(gids)
#   allocate_vector(V,ngids)
# end
#
# struct MPIPETScIJV{A,B}
#   dIJV::A
#   gIJV::B
# end
#
# get_distributed_data(a::MPIPETScIJV) = a.dIJV
#
# function Gridap.Algebra.allocate_coo_vectors(::Type{M},dn::DistributedData) where M <: AbstractMatrix
#
#   part_to_n = gather(dn)
#   n = sum(part_to_n)
#   gIJV = allocate_coo_vectors(M,n)
#
#   _fill_offsets!(part_to_n)
#   offsets = scatter(get_comm(dn),part_to_n.+1)
#
#   dIJV = DistributedData(offsets) do part, offset
#     map( i -> SubVector(i,offset,n), gIJV)
#   end
#
#   MPIPETScIJV(dIJV,gIJV)
# end
#
# function Gridap.Algebra.finalize_coo!(
#   ::Type{M},IJV::MPIPETScIJV,m::DistributedIndexSet,n::DistributedIndexSet) where M <: AbstractMatrix
#   I,J,V = IJV.gIJV
#   finalize_coo!(M,I,J,V,num_gids(m),num_gids(n))
# end
#
# function Gridap.Algebra.sparse_from_coo(
#   ::Type{M},IJV::MPIPETScIJV,m::DistributedIndexSet,n::DistributedIndexSet) where M
#   I,J,V = IJV.gIJV
#   sparse_from_coo(M,I,J,V,num_gids(m),num_gids(n))
# end
