
# A, B should be the type of some indexable collection, e.g. ranges or vectors
struct IndexSet{A,B}
  lid_to_gid::A
  lid_to_owner::B
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

struct SequentialDistributedIndexSet{A,B} <: DistributedIndexSet
  parts::SequentialDistributedData{IndexSet{A,B}}
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
