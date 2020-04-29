struct DistributedCellField
  cellfields::ScatteredVector{<:CellField}
end

function get_distributed_data(dcellfield::DistributedCellField)
  dcellfield.cellfields
end

function Gridap.Geometry.CellField(object,dtrian::DistributedTriangulation)
  cellfields = ScatteredVector(dtrian) do part, trian
    CellField(object,trian)
  end
  DistributedCellField(cellfields)
end

function Gridap.evaluate(dcellfield::DistributedCellField,dx)
  ScatteredVector(dcellfield,dx) do part, cellfield, x
    evaluate(cellfield,x)
  end
end
