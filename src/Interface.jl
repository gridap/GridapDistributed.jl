
abstract type DistributedGridapType <: GridapType end

function local_views(::DistributedGridapType)
  @abstractmethod
end

function get_parts(x::DistributedGridapType)
  return linear_indices(local_views(x))
end
