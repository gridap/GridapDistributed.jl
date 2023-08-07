
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

# i_am_in

function get_part_id(comm::MPI.Comm)
  if comm != MPI.COMM_NULL
    id = MPI.Comm_rank(comm)+1
  else
    id = -1
  end
  id
end

function i_am_in(comm::MPI.Comm)
  get_part_id(comm) >=0
end

function i_am_in(comm::MPIArray)
  i_am_in(comm.comm)
end

function i_am_in(comm::MPIVoidVector)
  i_am_in(comm.comm)
end

# change_parts

function change_parts(x::Union{MPIArray,DebugArray,Nothing,MPIVoidVector}, new_parts; default=nothing)
  x_new = map(new_parts) do _p
    if isa(x,MPIArray) || isa(x,DebugArray)
      PartitionedArrays.getany(x)
    else
      default
    end
  end
  return x_new
end
