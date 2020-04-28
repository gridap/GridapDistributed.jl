abstract type GhostedVector{T} end

Base.eltype(::Type{<:GhostedVector{T}}) where T = T
Base.eltype(::GhostedVector{T}) where T = T

get_part_type(::Type{<:GhostedVector{T}}) where T = GhostedVectorPart{T}
get_part_type(::GhostedVector{T}) where T = GhostedVectorPart{T}

# @santiagobadia : Think about the name... not sure ghosted meaning does have
# much sense in this context. GhostVector or something better (names in PETSc?)
# @santiagobadia : I think that the GhostedVectorPart should be abstract,
# I don't think we want these attributes for whatever GhostedVectorPart
# implementation.
# @santiagobadia : We should probably create a method that given an inconsistent
# GhostedVector provides a new consistent GhostedVector (after some comm
# nn comm algorithm) in the abstract interface instead. Or create a type that
# represents the vector without the comms. Do we want to do these operations in
# a lazy way? Does it have sense?

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
  parts::Vector{GhostedVectorPart{T}}
end

get_comm(a::SequentialGhostedVector) = SequentialCommunicator()

function GhostedVector{T}(
  initializer::Function,::SequentialCommunicator,nparts::Integer,args...) where T

  parts = [ initializer(i,map(a->a.parts[i],args)...) for i in 1:nparts ]
  SequentialGhostedVector{T}(parts)
end

function GhostedVector{T}(
  initializer::Function,a::SequentialGhostedVector,args...) where T

  nparts = length(a.parts)
  parts = [
    GhostedVectorPart(
    a.parts[i].ngids,
    initializer(i,map(a->a.parts[i],args)...),
    a.parts[i].lid_to_gid,
    a.parts[i].lid_to_owner,
    a.parts[i].gid_to_lid)
    for i in 1:nparts ]
  SequentialGhostedVector{T}(parts)
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

function GhostedVector{T}(initializer::Function,comm::MPICommunicator,nparts::Integer,args...) where T
  @assert nparts == num_parts(comm)
  largs = map(a->a.part,args)
  i = get_part(comm)
  part = initializer(i,largs...)
  MPIGhostedVector{T}(part,comm)
end


