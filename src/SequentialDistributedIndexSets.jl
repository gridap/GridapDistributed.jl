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
