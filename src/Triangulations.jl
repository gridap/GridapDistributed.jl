struct DistributedTriangulation{Dc,Dp,A,B}
  trians::A
  model::B
  function DistributedTriangulation(
    trians::AbstractPData{<:Triangulation{Dc,Dp}},
    model::DistributedDiscreteModel) where {Dc,Dp}
    A = typeof(trians)
    B = typeof(model)
    new{Dc,Dp,A,B}(trians,model)
  end
end

function Geometry.get_background_model(a::DistributedTriangulation)
  a.model
end

function Geometry.Triangulation(
  portion,model::DistributedDiscreteModel;kwargs...)
  trians = map_parts(model.models,model.gids.partition) do model,gids
    Triangulation(portion,gids,model;kwargs...)
  end
  DistributedTriangulation(trians,model)
end

function Geometry.BoundaryTriangulation(
  portion,model::DistributedDiscreteModel;kwargs...)
  trians = map_parts(model.models,model.gids.partition) do model,gids
    BoundaryTriangulation(portion,gids,model;kwargs...)
  end
  DistributedTriangulation(trians,model)
end

function Geometry.SkeletonTriangulation(
  portion,model::DistributedDiscreteModel;kwargs...)
  trians = map_parts(model.models,model.gids.partition) do model,gids
    SkeletonTriangulation(portion,gids,model;kwargs...)
  end
  DistributedTriangulation(trians,model)
end

function Geometry.Triangulation(
  portion,gids::AbstractIndexSet,args...;kwargs...)
  trian = Triangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.BoundaryTriangulation(
  portion,gids::AbstractIndexSet,args...;kwargs...)
  trian = BoundaryTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.SkeletonTriangulation(
  portion,gids::AbstractIndexSet,args...;kwargs...)
  trian = SkeletonTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.InterfaceTriangulation(
  portion,gids::AbstractIndexSet,args...;kwargs...)
  trian = InterfaceTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function filter_cells_when_needed(
  portion::PArrays.WithGhost,
  cell_gids::AbstractIndexSet,
  trian::Triangulation)

  trian
end

function filter_cells_when_needed(
  portion::PArrays.NoGhost,
  cell_gids::AbstractIndexSet,
  trian::Triangulation)

  remove_ghost_cells(trian,cell_gids)
end

# For the moment remove_ghost_cells
# refers to the triangulation faces
# pointing into the ghost cells
# in the underlying background discrete 
# model. This might change when solving
# multi-field PDEs with one of the fields
# defined on the boundary (e.g. a Lagrange multiplier)
function remove_ghost_cells(trian,gids)
  model = get_background_model(trian)
  D = num_cell_dims(model)
  glue = get_glue(trian,Val(D))
  remove_ghost_cells(glue,trian,gids)
end

function remove_ghost_cells(glue::FaceToFaceGlue,trian,gids)
  tcell_to_mcell = glue.tface_to_mface
  mcell_to_part = gids.lid_to_part
  tcell_to_part = view(mcell_to_part,tcell_to_mcell)
  tcell_to_mask = tcell_to_part .== gids.part
  view(trian, findall(tcell_to_mask))
end

function remove_ghost_cells(glue::SkeletonPair,trian,gids)
  plus = remove_ghost_cells(glue.plus,trian,gids)
  minus = remove_ghost_cells(glue.minus,trian,gids)
  SkeletonTriangulation(plus,minus)
end

