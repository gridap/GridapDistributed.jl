struct DistributedTriangulation
  trians::ScatteredVector{<:Triangulation}
end

function get_distributed_data(dtrian::DistributedTriangulation)
  dtrian.trians
end

function Gridap.writevtk(dtrian::DistributedTriangulation,filebase::String)

  do_on_parts(dtrian) do part, trian
    filebase_part = filebase*"_$(part)"
    writevtk(trian,filebase_part)
  end

end
