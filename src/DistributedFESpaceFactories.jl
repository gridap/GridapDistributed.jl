function Gridap.FESpace(::Type{V};
      model::DistributedDiscreteModel, reffe, constraint=nothing, kwargs...) where V
  if constraint == nothing
    DistributedFESpaceFromLocalFESpaces(V;model=model, reffe=reffe, kwargs...)
  elseif constraint == :zeromean
    ZeroMeanDistributedFESpace(V; model=model, reffe=reffe, kwargs...)
  else
    @unreachable "Unknown constraint value $constraint"
  end
end
