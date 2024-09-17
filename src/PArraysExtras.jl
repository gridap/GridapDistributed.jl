
"""
    permuted_variable_partition(n_own,gids,owners; n_global, start)

Create indices which are a permuted version of a variable_partition.
The advantage of this wrt the `LocalIndices` type, is that we can compute 
dof owners with minimal communications, since we only need the size of the blocks.

NOTE: This is the type for our FESpace dof_ids.
"""
function permuted_variable_partition(
  n_own::AbstractArray{<:Integer},
  gids::AbstractArray{<:AbstractArray{<:Integer}},
  owners::AbstractArray{<:AbstractArray{<:Integer}};
  n_global=reduction(+,n_own,destination=:all,init=zero(eltype(n_own))),
  start=scan(+,n_own,type=:exclusive,init=one(eltype(n_own)))
)
  ranks = linear_indices(n_own)
  np = length(ranks)
  map(ranks,n_own,n_global,start,gids,owners) do rank,n_own,n_global,start,gids,owners
    n_local = length(gids)
    n_ghost = n_local - n_own
    perm = fill(zero(Int32),n_local)
    ghost_gids = fill(zero(Int),n_ghost)
    ghost_owners = fill(zero(Int32),n_ghost)

    n_ghost = 0
    for (lid,(gid,owner)) in enumerate(zip(gids,owners))
      if owner == rank
        perm[lid] = gid-start+1
      else
        n_ghost += 1
        ghost_gids[n_ghost] = gid
        ghost_owners[n_ghost] = owner
        perm[lid] = n_own + n_ghost
      end
    end
    @assert n_ghost == n_local - n_own

    ghost = GhostIndices(n_global,ghost_gids,ghost_owners)
    dof_ids = PartitionedArrays.LocalIndicesWithVariableBlockSize(
      CartesianIndex((rank,)),(np,),(n_global,),((1:n_own).+(start-1),),ghost
    )
    permute_indices(dof_ids,perm)
  end
end

"""
    unpermute(indices::AbstractLocalIndices)

Given local indices, reorders them to (locally) have own indices first, 
followed by ghost indices.
"""
function unpermute(indices::AbstractLocalIndices) 
  @notimplemented
end

unpermute(indices::PermutedLocalIndices) = unpermute(indices.indices)
unpermute(indices::PartitionedArrays.LocalIndicesInBlockPartition) = indices
unpermute(indices::OwnAndGhostIndices) = indices

function unpermute(indices::LocalIndices)
  nglobal = global_length(indices)
  rank = part_id(indices)
  own = OwnIndices(nglobal,rank,own_to_global(indices))
  ghost = GhostIndices(nglobal,ghost_to_global(indices),ghost_to_owner(indices))
  OwnAndGhostIndices(own,ghost,global_to_owner(indices))
end

"""
    locally_repartition(v::PVector,new_indices)

Map the values of a PVector to a new partitioning of the indices.

Similar to `PartitionedArrays.repartition`, but without any communications. Instead, 
it is assumed that the local-to-local mapping can be done locally.
"""
function locally_repartition(v::PVector,new_indices)
  w = similar(v,PRange(new_indices))
  locally_repartition!(w,v)
end

function locally_repartition!(w::PVector,v::PVector)
  # Fill own values
  map(copy!,own_values(w),own_values(v))

  # Fill ghost values
  new_indices = partition(axes(w,1))
  old_indices = partition(axes(v,1))
  map(partition(w),partition(v),new_indices,old_indices) do w,v,new_ids,old_ids
    old_gid_to_lid = global_to_local(old_ids)
    for (lid,gid) in zip(ghost_to_local(new_ids),ghost_to_global(new_ids))
      old_lid = old_gid_to_lid[gid]
      w[lid] = v[old_lid]
    end
  end

  return w
end

# Linear algebra

function LinearAlgebra.axpy!(α,x::PVector,y::PVector)
  @check matching_local_indices(partition(axes(x,1)),partition(axes(y,1)))
  map(partition(x),partition(y)) do x,y
    LinearAlgebra.axpy!(α,x,y)
  end
  consistent!(y) |> wait
  return y
end

function LinearAlgebra.axpy!(α,x::BlockPVector,y::BlockPVector)
  map(blocks(x),blocks(y)) do x,y
    LinearAlgebra.axpy!(α,x,y)
  end
  return y
end

function Algebra.axpy_entries!(
  α::Number, A::PSparseMatrix, B::PSparseMatrix;
  check::Bool=true
)
# We should definitely check here that the index partitions are the same. 
# However: Because the different matrices are assembled separately, the objects are not the 
# same (i.e can't use ===). Checking the index partitions would then be costly...
  @assert reduce(&,map(PArrays.matching_local_indices,partition(axes(A,1)),partition(axes(B,1))))
  @assert reduce(&,map(PArrays.matching_local_indices,partition(axes(A,2)),partition(axes(B,2))))
  map(partition(A),partition(B)) do A, B
    Algebra.axpy_entries!(α,A,B;check)
  end
  return B
end

function Algebra.axpy_entries!(
  α::Number, A::BlockPMatrix, B::BlockPMatrix;
  check::Bool=true
)
  map(blocks(A),blocks(B)) do A, B
    Algebra.axpy_entries!(α,A,B;check)
  end
  return B
end

# Array of PArrays -> PArray of Arrays 
# TODO: I think this is now implemented in PartitionedArrays.jl (check)
function to_parray_of_arrays(a::AbstractArray{<:MPIArray})
  indices = linear_indices(first(a))
  map(indices) do i
    map(a) do aj
      PartitionedArrays.getany(aj)
    end
  end
end

function to_parray_of_arrays(a::AbstractArray{<:DebugArray})
  indices = linear_indices(first(a))
  map(indices) do i
    map(a) do aj
      aj.items[i]
    end
  end
end

# This type is required because MPIArray from PArrays 
# cannot be instantiated with a NULL communicator
struct MPIVoidVector{T} <: AbstractVector{T}
  comm::MPI.Comm
  function MPIVoidVector(::Type{T}) where {T}
    new{T}(MPI.COMM_NULL)
  end
end

Base.size(a::MPIVoidVector) = (0,)
Base.IndexStyle(::Type{<:MPIVoidVector}) = IndexLinear()
function Base.getindex(a::MPIVoidVector,i::Int)
  error("Indexing of MPIVoidVector not possible.")
end
function Base.setindex!(a::MPIVoidVector,v,i::Int)
  error("Indexing of MPIVoidVector not possible.")
end
function Base.show(io::IO,k::MIME"text/plain",data::MPIVoidVector)
  println(io,"MPIVoidVector")
end

# Communication extras, subpartitioning extras

function num_parts(comm::MPI.Comm)
  if comm != MPI.COMM_NULL
    nparts = MPI.Comm_size(comm)
  else
    nparts = -1
  end
  nparts
end
@inline num_parts(comm::MPIArray) = num_parts(comm.comm)
@inline num_parts(comm::DebugArray) = length(comm.items)
@inline num_parts(comm::MPIVoidVector) = num_parts(comm.comm)

function get_part_id(comm::MPI.Comm)
  if comm != MPI.COMM_NULL
    id = MPI.Comm_rank(comm)+1
  else
    id = -1
  end
  id
end
@inline get_part_id(comm::MPIArray) = get_part_id(comm.comm)
@inline get_part_id(comm::MPIVoidVector) = get_part_id(comm.comm)

"""
    i_am_in(comm::MPIArray)
    i_am_in(comm::DebugArray)
  
  Returns `true` if the processor is part of the subcommunicator `comm`.
"""
function i_am_in(comm::MPI.Comm)
  get_part_id(comm) >=0
end
@inline i_am_in(comm::MPIArray) = i_am_in(comm.comm)
@inline i_am_in(comm::MPIVoidVector) = i_am_in(comm.comm)
@inline i_am_in(comm::DebugArray) = true

function change_parts(x::Union{MPIArray,DebugArray,Nothing,MPIVoidVector}, new_parts; default=nothing)
  x_new = map(new_parts) do p
    if isa(x,MPIArray)
      PartitionedArrays.getany(x)
    elseif isa(x,DebugArray) && (p <= length(x.items))
      x.items[p]
    else
      default
    end
  end
  return x_new
end

function generate_subparts(parts::MPIArray,new_comm_size)
  root_comm = parts.comm
  root_size = MPI.Comm_size(root_comm)
  rank = MPI.Comm_rank(root_comm)

  @static if isdefined(MPI,:MPI_UNDEFINED)
    mpi_undefined = MPI.MPI_UNDEFINED[]
  else
    mpi_undefined = MPI.API.MPI_UNDEFINED[]
  end
  
  if root_size == new_comm_size
    return parts
  else
    if rank < new_comm_size
      comm = MPI.Comm_split(root_comm,0,0)
      return distribute_with_mpi(LinearIndices((new_comm_size,));comm=comm,duplicate_comm=false)
    else
      comm = MPI.Comm_split(root_comm,mpi_undefined,mpi_undefined)
      return MPIVoidVector(eltype(parts))
    end
  end
end

function generate_subparts(parts::DebugArray,new_comm_size)
  DebugArray(LinearIndices((new_comm_size,)))
end

