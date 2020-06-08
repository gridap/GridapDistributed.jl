# Specializations
struct MPIPETScDistributedVector{T<:Number,V<:AbstractVector{T},A,B,C}
  part      :: V
  indices   :: MPIPETScDistributedIndexSet{A,B,C}
  vecghost  :: PETSc.Vec{Float64}
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

function DistributedVector{T}(
  initializer::Function, indices::MPIPETScDistributedIndexSet,args...) where T
  comm = get_comm(indices)
  data = DistributedData(initializer,comm,args...)
  part = data.part
  println(data.part)
  @assert sizeof(eltype(part)) == sizeof(Float64)
  @assert length(part)         == length(indices.parts.part.lid_to_owner)
  vecghost = _create_ghost_vector(indices)
  MPIPETScDistributedVector{T}(part,indices,vecghost)
end

function DistributedVector(
  initializer::Function, indices::MPIPETScDistributedIndexSet,args...)
  comm = get_comm(indices)
  data = DistributedData(initializer,comm,args...)
  part = data.part
  @assert sizeof(eltype(part)) == sizeof(Float64)
  @assert length(part)         == length(indices.parts.part.lid_to_owner)
  vecghost = _create_ghost_vector(indices)
  MPIPETScDistributedVector(part,indices,vecghost)
end

function _create_ghost_vector(indices::MPIPETScDistributedIndexSet)
  comm = get_comm(indices)
  comm_rank = MPI.Comm_rank(comm.comm)
  ghost_idx=Int[]
  lid_to_owner = indices.parts.part.lid_to_owner
  lid_to_gid_petsc  = indices.lid_to_gid_petsc
  num_owned_entries = _num_owned_entries(indices)
  num_local_entries = length(lid_to_owner)
  for i=1:num_local_entries
    if (lid_to_owner[i]!==comm_rank+1)
       push!(ghost_idx, lid_to_gid_petsc[i])
    end
  end
  VecGhost(Float64, num_owned_entries, ghost_idx)
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
  local_part    = a.part
  petsc_to_app_locidx = indices.petsc_to_app_locidx
  app_to_petsc_locidx = indices.app_to_petsc_locidx

  # Pack data
  num_owned_entries = _num_owned_entries(indices)
  num_local_entries = length(petsc_to_app_locidx)
  lvecghost = PETSc.VecLocal(a.vecghost)
  lvec      = PETSc.LocalVector(lvecghost)
  for i=1:num_owned_entries
    lvec.a[i]=reinterpret(Float64,local_part[petsc_to_app_locidx[i]])
  end
  PETSc.restore(lvec)
  PETSc.restore(lvecghost)

  # Send data
  PETSc.scatter!(a.vecghost)

  # Unpack data
  comm = get_comm(indices)
  comm_rank = MPI.Comm_rank(comm.comm)
  lid_to_owner = indices.parts.part.lid_to_owner

  lvecghost = PETSc.VecLocal(a.vecghost)
  lvec      = PETSc.LocalVector(lvecghost)
  for i=1:num_local_entries
    if (lid_to_owner[i] !== comm_rank+1)
      local_part[i]=reinterpret(T,lvec.a[app_to_petsc_locidx[i]])
    end
  end
  PETSc.restore(lvec)
  PETSc.restore(lvecghost)
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
