struct DistributedTriangulation
  trians::ScatteredVector{<:Triangulation}
end

function get_distributed_data(dtrian::DistributedTriangulation)
  dtrian.trians
end

function Gridap.writevtk(dtrian::DistributedTriangulation,filebase::String;cellfields=Dict())

  d = Dict(cellfields)
  thevals = values(d)
  thekeys = keys(d)

  do_on_parts(dtrian,thevals...) do part, trian, thevals...
    filebase_part = filebase*"_$(part)"
    cellfields = [ k=>v for (k,v) in zip(thekeys, thevals) ]
    writevtk(trian,filebase_part,cellfields=cellfields)
  end

end
