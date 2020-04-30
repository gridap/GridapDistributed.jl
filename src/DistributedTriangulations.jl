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

#
#function Gridap.Triangulation(dmodel::DistributedDiscreteModel,args...)
#  comm = get_comm(dmodel)
#  trians = ScatteredVector(comm,dmodel.models) do part, model
#    Triangulation(model,args...)
#  end
#  DistributedTriangulation(trians)
#end
#
#function Gridap.BoundaryTriangulation(dmodel::DistributedDiscreteModel,args...)
#  comm = get_comm(dmodel)
#  trians = ScatteredVector(comm,dmodel.models) do part, model
#    BoundaryTriangulation(model,args...)
#  end
#  DistributedTriangulation(trians)
#end
#
#function Gridap.SkeletonTriangulation(dmodel::DistributedDiscreteModel,args...)
#  comm = get_comm(dmodel)
#  trians = ScatteredVector(comm,dmodel.models) do part, model
#    SkeletonTriangulation(model,args...)
#  end
#  DistributedTriangulation(trians)
#end
#
#
#function remove_ghost_cells(dtrian::DistributedTriangulation,dmodel)
#
#  trians = ScatteredVector(dtrian,dmodel.gids) do part, trian, gids
#
#    tcell_to_mcell = get_cell_id(trian)
#    mcell_to_isowned = gids.lid_to_owner .== part
#    tcell_to_isowned = reindex(mcell_to_isowned,tcell_to_mcell)
#    ocell_to_tcell = findall(tcell_to_isowned)
#    TriangulationPortion(trian,ocell_to_tcell)
#  end
#
#  DistributedTriangulation(trians)
#
#end
#
#function include_ghost_cells(dtrian::DistributedTriangulation)
#
#  trians = ScatteredVector(dtrian) do part, trian
#    trian.oldtrian
#  end
#
#  DistributedTriangulation(trians)
#end
#
##TODO move to Gridap
#
#function Gridap.Geometry.get_cell_id(trian::TriangulationPortion)
#  reindex(get_cell_id(trian.oldtrian),trian.cell_to_oldcell)
#end
