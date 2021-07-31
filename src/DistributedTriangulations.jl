abstract type ProcLocalTriangulationPortion end;
struct OwnedCells         <: ProcLocalTriangulationPortion end;
struct OwnedAndGhostCells <: ProcLocalTriangulationPortion end;

function filter_cells_when_needed(portion::ProcLocalTriangulationPortion,
                                  trian::Triangulation,
                                  args...)
   filter_cells_when_needed(typeof(portion), trian, args...)
end

function filter_cells_when_needed(portion::Type{<:ProcLocalTriangulationPortion},
                                  trian::Triangulation,
                                  args...)
  @abstractmethod
end

function filter_cells_when_needed(portion::Type{OwnedAndGhostCells},
                                  trian::Triangulation,
                                  args...)
  trian
end

function filter_cells_when_needed(portion::Type{OwnedCells},
                                  trian::Triangulation,
                                  part,
                                  cell_gids)
  remove_ghost_cells(trian,part,cell_gids)
end

function Gridap.Geometry.Triangulation(portion::Type{<:ProcLocalTriangulationPortion},
  part,gids,model::DiscreteModel,args_triangulation...)#;kwargs_triangulation...)
  @abstractmethod
end

function Gridap.Geometry.Triangulation(portion::Type{<:ProcLocalTriangulationPortion},
  model::DistributedDiscreteModel,args_triangulation...)#;kwargs_triangulation...)
  DistributedData(model) do part, (model,gids)
    Triangulation(portion,part,gids,model,args_triangulation...)
  end
end

function Gridap.Geometry.Triangulation(portion::Type{OwnedCells},
  part,gids,model::DiscreteModel,args_triangulation...)#;kwargs_triangulation...)
  trian=Triangulation(model,args_triangulation...)#;kwargs_triangulation...)
  filter_cells_when_needed(portion,trian,part,gids)
end

function Gridap.Geometry.Triangulation(portion::Type{OwnedAndGhostCells},
  part,gids,model::DiscreteModel,args_triangulation...)#;kwargs_triangulation...)
  trian=Triangulation(model,args_triangulation...)#;kwargs_triangulation...)
  filter_cells_when_needed(portion,trian)
end

#TO-DO: what does it mean OwnedCells and OwnedAndGhostCells for a BoundaryTriangulation?
#       Perhaps we should use a different type, with a more intention revealing name!
function Gridap.Geometry.BoundaryTriangulation(portion::Type{<:ProcLocalTriangulationPortion},
  part,gids,model::DiscreteModel;kwargs_triangulation...)
  @abstractmethod
end

function Gridap.Geometry.BoundaryTriangulation(portion::Type{<:ProcLocalTriangulationPortion},
  model::DistributedDiscreteModel;kwargs_triangulation...)
  DistributedData(model) do part, (model,gids)
    BoundaryTriangulation(portion,part,gids,model;kwargs_triangulation...)
  end
end

function Gridap.Geometry.BoundaryTriangulation(portion::Type{OwnedCells},
  part,gids,model::DiscreteModel;kwargs_triangulation...)
  trian=BoundaryTriangulation(model;kwargs_triangulation...)
  filter_cells_when_needed(portion,trian,part,gids)
end

function Gridap.Geometry.BoundaryTriangulation(portion::Type{OwnedAndGhostCells},
  part,gids,model::DiscreteModel;kwargs_triangulation...)
  trian=BoundaryTriangulation(model;kwargs_triangulation...)
  filter_cells_when_needed(portion,trian)
end

#TO-DO: what does it mean OwnedCells and OwnedAndGhostCells for a SkeletonTriangulation?
#       Perhaps we should use a different type, with a more intention revealing name!
function Gridap.Geometry.SkeletonTriangulation(portion::Type{OwnedCells},
  model::DistributedDiscreteModel;kwargs_triangulation...)
  @notimplemented
end

function Gridap.Geometry.SkeletonTriangulation(portion::Type{OwnedAndGhostCells},
  model::DistributedDiscreteModel,args_triangulation...;kwargs_triangulation...)
  @notimplemented
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
