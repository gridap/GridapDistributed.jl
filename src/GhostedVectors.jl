abstract type GhostedVector{T} <: DistributedData end

Base.eltype(::Type{<:GhostedVector{T}}) where T = T
Base.eltype(::GhostedVector{T}) where T = T

get_part_type(::Type{<:GhostedVector{T}}) where T = GhostedVectorPart{T}
get_part_type(::GhostedVector{T}) where T = GhostedVectorPart{T}

# @santiagobadia : Renaming after discussion, in which GhostedVector is a
# DistributedIndexSet(and _Part) and then a DistributedVector(and _Part).
# Consider an abstract version of these structs. E.g., I don't think we want
# these things to be Vector or Dict, instead they could be an AbstractVector
# and create lazy vectors such that get_index provides fun(i) instead, which is
# going to be useful in many practical implementations. E.g., gid_to_lid(gi) =
# gi - offset, lid_to_gid(gi) = li + offset for owned vefs, we could define
# lid_to_gid based on in which rank i falls, etc.
# @santiagobadia : The interface will certainly require more methods after
# the changes above
# @santiagobadia : The exchange! method sends and receives data from other
# procs, would it have sense a two-stage approach too? Send and only receive
# when needed?
struct GhostedVectorPart{T}
  ngids::Int
  lid_to_item::Vector{T}
  lid_to_gid::Vector{Int}
  lid_to_owner::Vector{Int}
  gid_to_lid::Dict{Int,Int32}
end

function GhostedVectorPart{T}(
  ngids::Int,
  lid_to_item::Vector,
  lid_to_gid::Vector{Int},
  lid_to_owner::Vector{Int}) where T

  gid_to_lid = Dict{Int,Int32}()
  for (lid,gid) in enumerate(lid_to_gid)
    gid_to_lid[gid] = lid
  end
  GhostedVectorPart{T}(
    ngids,
    lid_to_item,
    lid_to_gid,
    lid_to_owner,
    gid_to_lid)
end

function GhostedVectorPart(
  ngids::Int,
  lid_to_item::Vector{T},
  lid_to_gid::Vector{Int},
  lid_to_owner::Vector{Int}) where T

  GhostedVectorPart{T}(
    ngids,
    lid_to_item,
    lid_to_gid,
    lid_to_owner)
end

function get_comm(::GhostedVector)
  @abstractmethod
end

function exchange!(::GhostedVector)
  @abstractmethod
end

function GhostedVector{T}(
  initializer::Function,::Communicator,nparts::Integer,args...) where T
  @abstractmethod
end

function GhostedVector{T}(
  initializer::Function,::GhostedVector,args...) where T
  @abstractmethod
end

struct SequentialGhostedVector{T} <: GhostedVector{T}
  comm::SequentialCommunicator
  parts::Vector{GhostedVectorPart{T}}
end

get_comm(a::SequentialGhostedVector) = a.comm

num_parts(a::SequentialGhostedVector) = length(a.parts)

function GhostedVector{T}(
  initializer::Function,comm::SequentialCommunicator,nparts::Integer,args...) where T

  parts = [ initializer(i,map(a->get_distributed_data(a).parts[i],args)...) for i in 1:nparts ]
  SequentialGhostedVector{T}(comm,parts)
end

function GhostedVector{T}(
  initializer::Function,a::SequentialGhostedVector,args...) where T

  nparts = length(a.parts)
  parts = [
    GhostedVectorPart(
    a.parts[i].ngids,
    initializer(i,map(a->get_distributed_data(a).parts[i],args)...),
    a.parts[i].lid_to_gid,
    a.parts[i].lid_to_owner,
    a.parts[i].gid_to_lid)
    for i in 1:nparts ]
  SequentialGhostedVector{T}(a.comm,parts)
end

function exchange!(a::SequentialGhostedVector)
  for part in 1:length(a.parts)
    lid_to_gid = a.parts[part].lid_to_gid
    lid_to_item = a.parts[part].lid_to_item
    lid_to_owner = a.parts[part].lid_to_owner
    for lid in 1:length(lid_to_item)
      gid = lid_to_gid[lid]
      owner = lid_to_owner[lid]
      if owner != part
        lid_owner = a.parts[owner].gid_to_lid[gid]
        item = a.parts[owner].lid_to_item[lid_owner]
        lid_to_item[lid] = item
      end
    end
  end
end

struct MPIGhostedVector{T} <: GhostedVector{T}
  part::GhostedVectorPart{T}
  comm::MPICommunicator
end

get_comm(a::MPIGhostedVector) = a.comm

num_parts(a::MPIGhostedVector) = num_parts(a.comm)

function GhostedVector{T}(initializer::Function,comm::MPICommunicator,nparts::Integer,args...) where T
  @assert nparts == num_parts(comm)
  largs = map(a->get_distributed_data(a).part,args)
  i = get_part(comm)
  part = initializer(i,largs...)
  MPIGhostedVector{T}(part,comm)
end
