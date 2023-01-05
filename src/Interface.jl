
abstract type DistributedGridapType <: GridapType end

function local_views(::DistributedGridapType)
  @abstractmethod
end

function get_parts(x::DistributedGridapType)
  return PArrays.get_part_ids(local_views(x))
end
