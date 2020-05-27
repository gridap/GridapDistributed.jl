
function filter_cells_when_needed(strategy::AssemblyStrategy, model::DiscreteModel, trian::Triangulation)
  @abstractmethod
end

function filter_cells_when_needed(strategy::RowsComputedLocally, model::DiscreteModel, trian::Triangulation)
  trian
end

function filter_cells_when_needed(strategy::OwnedCellsStrategy, model::DiscreteModel, trian::Triangulation)
  remove_ghost_cells(trian,strategy.part,strategy.cell_gids)
end

function filter_cells_when_needed(strategy::OwnedCellsStrategy, model::DiscreteModel, trian::SkeletonTriangulation)
  topo = get_grid_topology(model)
  D = num_cell_dims(model)
  facets_to_cells = Gridap.Geometry.get_faces(topo,D-1,D)
  facet_lids = Gridap.Geometry.get_cell_id(trian.left.face_trian)
  facets_to_old_facets = eltype(facet_lids)[]
  for (i,facet_lid) in enumerate(facet_lids)
    max_part_id=-1
    for j=facets_to_cells.ptrs[facet_lid]:facets_to_cells.ptrs[facet_lid+1]-1
       cell_lid  = facets_to_cells.data[j]
       cell_part = strategy.cell_gids.lid_to_owner[cell_lid]
       max_part_id=max(max_part_id, cell_part)
    end
    if (strategy.part == max_part_id)
      push!(facets_to_old_facets, i)
    end
  end
  TriangulationPortion(trian,facets_to_old_facets)
  #trian
end

function Gridap.Geometry.Triangulation(strategy::AssemblyStrategy,model::DiscreteModel,args...)
  trian = Triangulation(model,args...)
  filter_cells_when_needed(strategy,model,trian)
end

function Gridap.Geometry.BoundaryTriangulation(strategy::AssemblyStrategy,model::DiscreteModel,args...)
  trian = BoundaryTriangulation(model,args...)
  filter_cells_when_needed(strategy,model,trian)
end

function Gridap.Geometry.SkeletonTriangulation(strategy::AssemblyStrategy,model::DiscreteModel,args...)
  trian = SkeletonTriangulation(model,args...)
  filter_cells_when_needed(strategy,model,trian)
end

function remove_ghost_cells(trian::Triangulation,part::Integer,gids::IndexSet)
    tcell_to_mcell = get_cell_id(trian)
    mcell_to_isowned = gids.lid_to_owner .== part
    tcell_to_isowned = reindex(mcell_to_isowned,tcell_to_mcell)
    ocell_to_tcell = findall(tcell_to_isowned)
    TriangulationPortion(trian,ocell_to_tcell)
end

function include_ghost_cells(trian::TriangulationPortion)
    trian.oldtrian
end
