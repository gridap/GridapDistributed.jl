struct DistributedTriangulation
  trians::ScatteredVector{<:Triangulation}
end

function get_distributed_data(dtrian::DistributedTriangulation)
  dtrian.trians
end

function Gridap.Triangulation(dmodel::DistributedDiscreteModel,args...)
  comm = get_comm(dmodel)
  nparts = num_parts(dmodel)
  trians = ScatteredVector{Triangulation}(comm,nparts,dmodel.models) do part, model
    Triangulation(model,args...)
  end
  DistributedTriangulation(trians)
end

function Gridap.BoundaryTriangulation(dmodel::DistributedDiscreteModel,args...)
  comm = get_comm(dmodel)
  nparts = num_parts(dmodel)
  trians = ScatteredVector{Triangulation}(comm,nparts,dmodel.models) do part, model
    BoundaryTriangulation(model,args...)
  end
  DistributedTriangulation(trians)
end

function Gridap.SkeletonTriangulation(dmodel::DistributedDiscreteModel,args...)
  comm = get_comm(dmodel)
  nparts = num_parts(dmodel)
  trians = ScatteredVector{Triangulation}(comm,nparts,dmodel.models) do part, model
    SkeletonTriangulation(model,args...)
  end
  DistributedTriangulation(trians)
end

function Gridap.writevtk(dtrian::DistributedTriangulation,filebase::String)

  do_on_parts(dtrian) do part, trian
    filebase_part = filebase*"_$(part)"
    writevtk(trian,filebase_part)
  end

end
