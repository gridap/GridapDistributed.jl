
function filter_cells_when_needed(strategy::AssemblyStrategy, trian::Triangulation)
  @abstractmethod
end

function filter_cells_when_needed(strategy::RowsComputedLocally, trian::Triangulation)
  trian
end

function filter_cells_when_needed(strategy::OwnedCellsStrategy, trian::Triangulation)
  remove_ghost_cells(trian,strategy.part,strategy.cell_gids)
end

function Gridap.Geometry.Triangulation(strategy::AssemblyStrategy,model::DiscreteModel,args...)
  trian = Triangulation(model,args...)
  filter_cells_when_needed(strategy,trian)
end

function Gridap.Geometry.BoundaryTriangulation(strategy::AssemblyStrategy,model::DiscreteModel,args...)
  trian = BoundaryTriangulation(model,args...)
  filter_cells_when_needed(strategy,trian)
end

function Gridap.Geometry.SkeletonTriangulation(strategy::AssemblyStrategy,model::DiscreteModel,args...)
  trian = SkeletonTriangulation(model,args...)
  filter_cells_when_needed(strategy,trian)
end

function remove_ghost_cells(trian::Triangulation, part::Integer, gids::IndexSet)
    tcell_to_mcell = get_cell_id(trian)
    ocell_to_tcell =
        findall((x) -> (gids.lid_to_owner[x] == part), tcell_to_mcell)
    # TO-DO: TriangulationPortion(trian, ocell_to_tcell)
end

function remove_ghost_cells(
    trian::SkeletonTriangulation,
    part::Integer,
    gids::IndexSet,
)
    cell_id_left = get_cell_id(trian.left)
    cell_id_right = get_cell_id(trian.right)
    @assert length(cell_id_left) == length(cell_id_right)
    facets_to_old_facets =
        _compute_facets_to_old_facets(cell_id_left, cell_id_right, part, gids)
    # TO-DO: TriangulationPortion(trian, facets_to_old_facets)
end

function _compute_facets_to_old_facets(cell_id_left, cell_id_right, part, gids)
    facets_to_old_facets = eltype(cell_id_right)[]
    for i = 1:length(cell_id_left)
        part_left = gids.lid_to_owner[cell_id_left[i]]
        part_right = gids.lid_to_owner[cell_id_right[i]]
        max_part_id = max(part_left, part_right)
        if (max_part_id == part)
            push!(facets_to_old_facets, i)
        end
    end
    facets_to_old_facets
end

function include_ghost_cells(trian) # TO-DO :: TriangulationPortion)
    trian.oldtrian
end
