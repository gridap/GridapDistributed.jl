
struct VoidDistributedFESpace{A} <: Gridap.GridapType
  parts::A
end

function change_parts(x::Union{MPIArray,DebugArray,Nothing}, new_parts; default=nothing)
  x_new = map(new_parts) do _p
    if isa(x,MPIArray) || isa(x,DebugArray)
      PartitionedArrays.getany(x)
    else
      default
    end
  end
  return x_new
end
