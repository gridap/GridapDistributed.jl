
function Gridap.Geometry.Triangulation(model::DistributedDiscreteModel)
  das=default_assembly_strategy_type(get_comm(model))
  Triangulation(das,model)
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

function (*)(a::Gridap.CellData.Integrand,b::DistributedData{<:Measure})
  integrate(a.object,b)
end

(*)(b::DistributedData{<:Measure},a::Gridap.CellData.Integrand) = a*b

function Gridap.CellData.integrate(f::DistributedData,b::DistributedData{<:Measure})
  DistributedData(f,b) do part,f,b
    integrate(f,b)
  end
end

function Base.sum(a::DistributedData{<:Gridap.CellData.DomainContribution})
   g=DistributedData(a) do part, a
      sum(a)
   end
   sum(gather(g))
end
