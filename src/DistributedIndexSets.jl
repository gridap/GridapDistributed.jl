
# A, B, C should be the type of some indexable collection, e.g. ranges or vectors or dicts
struct IndexSet{A,B,C}
  ngids::Int
  lid_to_gid::A
  lid_to_owner::B
  gid_to_lid::C
end

function IndexSet(ngids,lid_to_gid,lid_to_owner)
  gid_to_lid = Dict{Int,Int32}()
  for (lid,gid) in enumerate(lid_to_gid)
    gid_to_lid[gid] = lid
  end
  IndexSet(ngids,lid_to_gid,lid_to_owner,gid_to_lid)
end

abstract type DistributedIndexSet end

function num_gids(a::DistributedIndexSet)
  @abstractmethod
end

function get_comm(a::DistributedIndexSet)
  @abstractmethod
end

function DistributedIndexSet(initializer::Function,::Communicator,ngids::Integer,args...)
  @abstractmethod
end

# The comm argument can be omitted if it can be determined from the first
# data argument.
function DistributedIndexSet(initializer::Function,ngids::Integer,args...) where T
  comm = get_comm(get_distributed_data(first(args)))
  DistributedIndexSet(initializer,comm,args...)
end

# Specializations

struct SequentialDistributedIndexSet{A,B,C} <: DistributedIndexSet
  ngids::Int
  parts::SequentialDistributedData{IndexSet{A,B,C}}
end

num_gids(a::SequentialDistributedIndexSet) = a.ngids

get_part(
  comm::SequentialCommunicator,
  a::SequentialDistributedIndexSet,
  part::Integer) = a.parts.parts[part]

get_comm(a::SequentialDistributedIndexSet) = a.parts.comm

function DistributedIndexSet(initializer::Function,comm::SequentialCommunicator,ngids::Integer,args...)
  parts = DistributedData(initializer,comm,args...)
  SequentialDistributedIndexSet(ngids,parts)
end

