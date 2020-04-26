
abstract type GhostedVector{T} end

struct GhostedVectorPart{T}
  lid_to_item::Vector{T}
  lid_to_gid::Vector{Int}
  lid_to_owner::Vector{Int}
  gid_to_lid::Dict{Int,Int32}
end

function GhostedVectorPart{T}(
  lid_to_item::Vector,
  lid_to_gid::Vector{Int},
  lid_to_owner::Vector{Int}) where T
  
  gid_to_lid = Dict{Int,Int32}()
  for (lid,gid) in enumerate(lid_to_gid)
    gid_to_lid[gid] = lid
  end
  GhostedVectorPart{T}(
    lid_to_item,
    lid_to_gid,
    lid_to_owner,
    gid_to_lid)
end

function GhostedVectorPart(
  lid_to_item::Vector{T},
  lid_to_gid::Vector{Int},
  lid_to_owner::Vector{Int}) where T

  GhostedVectorPart{T}(
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
  ::Communicator,nparts::Integer,initializer::Function,args...) where T
  @abstractmethod
end

function GhostedVector{T}(
  ::GhostedVector,initializer::Function,args...) where T
  @abstractmethod
end

struct SequentialGhostedVector{T} <: GhostedVector{T}
  parts::Vector{GhostedVectorPart{T}}
end

get_comm(a::SequentialGhostedVector) = SequentialCommunicator()

function GhostedVector{T}(
  ::SequentialCommunicator,nparts::Integer,initializer::Function,args...) where T

  parts = [ initializer(i,map(a->a.parts[i],args)...) for i in 1:nparts ]
  SequentialGhostedVector{T}(parts)
end

function GhostedVector{T}(
  a::SequentialGhostedVector,initializer::Function,args...) where T

  nparts = length(a.parts)
  parts = [
    GhostedVectorPart(
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


