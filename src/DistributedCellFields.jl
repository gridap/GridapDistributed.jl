struct DistributedCellField
  cellfields::ScatteredVector{<:CellField}
end

function get_distributed_data(dcellfield::DistributedCellField)
  dcellfield.cellfields
end

function Gridap.Geometry.CellField(object,dtrian::DistributedTriangulation)
  #TODO CellField
  cellfields = ScatteredVector{CellField}(dtrian) do part, trian
    CellField(object,trian)
  end
  DistributedCellField(cellfields)
end

function Gridap.evaluate(dcellfield::DistributedCellField,dx)
  #TODO Any
  ScatteredVector{Any}(dcellfield,dx) do part, cellfield, x
    evaluate(cellfield,x)
  end
end
