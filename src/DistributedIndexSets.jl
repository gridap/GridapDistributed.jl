
# A, B, C should be the type of some indexable collection, e.g. ranges or vectors or dicts
struct IndexSet{A,B,C}
  lid_to_gid::A
  lid_to_owner::B
  gid_to_lid::C
end

function IndexSet(lid_to_gid,lid_to_owner)
  gid_to_lid = Dict{Int,Int32}()
  for (lid,gid) in enumerate(lid_to_gid)
    gid_to_lid[gid] = lid
  end
  IndexSet(lid_to_gid,lid_to_owner,gid_to_lid)
end

abstract type DistributedIndexSet end

function get_comm(a::DistributedIndexSet)
  @abstractmethod
end

function DistributedIndexSet(initializer::Function,::Communicator,args...)
  @abstractmethod
end

# The comm argument can be omitted if it can be determined from the first
# data argument.
function DistributedIndexSet(initializer::Function,args...) where T
  comm = get_comm(get_distributed_data(first(args)))
  DistributedIndexSet(initializer,comm,args...)
end

# Specializations

struct SequentialDistributedIndexSet{A,B,C} <: DistributedIndexSet
  parts::SequentialDistributedData{IndexSet{A,B,C}}
end

get_part(
  comm::SequentialCommunicator,
  a::SequentialDistributedIndexSet,
  part::Integer) = a.parts.parts[part]

get_comm(a::SequentialDistributedIndexSet) = a.parts.comm

function DistributedIndexSet(initializer::Function,comm::SequentialCommunicator,args...)
  parts = DistributedData(initializer,comm,args...)
  SequentialDistributedIndexSet(parts)
end
