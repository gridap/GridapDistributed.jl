function Gridap.FESpace(::Type{V}; model::DistributedDiscreteModel,kwargs...) where V
  constraint=Gridap.FESpaces._get_kwarg(:constraint,kwargs,nothing)
  if constraint == nothing
    DistributedFESpaceFromLocalFESpaces(V;model=model,kwargs...)
  elseif constraint == :zeromean
    dkwargs=Dict(kwargs)
    delete!(dkwargs,:constraint)
    ZeroMeanDistributedFESpace(V;model=model,dkwargs...)
  else
    @unreachable "Unknown constraint value $constraint"
  end
end
