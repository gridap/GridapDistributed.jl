
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

function Gridap.Geometry.BoundaryTriangulation(
  strategy::AssemblyStrategy,
  model::DiscreteModel, args...;kwargs...)
  trian = BoundaryTriangulation(model,args...;kwargs...)
  filter_cells_when_needed(strategy,trian)
end

function Gridap.Geometry.SkeletonTriangulation(strategy::AssemblyStrategy,model::DiscreteModel,args...)
  trian = SkeletonTriangulation(model,args...)
  filter_cells_when_needed(strategy,trian)
end

function remove_ghost_cells(trian::Triangulation, part::Integer, gids::IndexSet)
    tcell_to_mcell = get_cell_to_bgcell(trian)
    ocell_to_tcell =
        findall((x) -> (gids.lid_to_owner[x] == part), tcell_to_mcell)
    RestrictedTriangulation(trian, ocell_to_tcell)
end

function remove_ghost_cells(
    trian::SkeletonTriangulation,
    part::Integer,
    gids::IndexSet,
)
    cell_id_plus = get_cell_to_bgcell(trian.plus)
    cell_id_minus = get_cell_to_bgcell(trian.minus)
    @assert length(cell_id_plus) == length(cell_id_minus)
    facets_to_old_facets =
        _compute_facets_to_old_facets(cell_id_plus, cell_id_minus, part, gids)
    RestrictedTriangulation(trian, facets_to_old_facets)
end

function _compute_facets_to_old_facets(cell_id_plus, cell_id_minus, part, gids)
    facets_to_old_facets = eltype(cell_id_minus)[]
    for i = 1:length(cell_id_plus)
        part_plus = gids.lid_to_owner[cell_id_plus[i]]
        part_minus = gids.lid_to_owner[cell_id_minus[i]]
        max_part_id = max(part_plus, part_minus)
        if (max_part_id == part)
            push!(facets_to_old_facets, i)
        end
    end
    facets_to_old_facets
end

function include_ghost_cells(trian::RestrictedTriangulation)
    trian.oldtrian
end
