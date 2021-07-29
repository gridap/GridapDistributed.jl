
function Gridap.Geometry.Triangulation(model::DistributedDiscreteModel)
  das=default_assembly_strategy_type(get_comm(model))
  Triangulation(das,model)
end

function Gridap.Geometry.Triangulation(
  model::DistributedDiscreteModel,strategy::DistributedAssemblyStrategy)

end

function Gridap.CellData.CellQuadrature(trian::DistributedData{<:Triangulation},degree::Integer)
  DistributedData(trian) do part, trian
    cell_quad = Gridap.CellData.Quadrature(trian,degree)
    CellQuadrature(trian,cell_quad)
  end
end

function Gridap.CellData.Measure(trian::DistributedData{<:Triangulation},degree::Integer)
  cell_quad=Gridap.CellData.CellQuadrature(trian,degree)
  DistributedData(cell_quad) do part, cell_quad
    Measure(cell_quad)
  end
end
