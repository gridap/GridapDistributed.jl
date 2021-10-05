
abstract type DistributedCellDatum end

CellData.get_data(a::DistributedCellDatum) = @abstractmethod
CellData.DomainStyle(::Type{<:DistributedCellDatum}) = @abstractmethod
CellData.get_triangulation(a::DistributedCellDatum) = @abstractmethod
CellData.change_domain(a::DistributedCellDatum,target_domain::DomainStyle) = change_domain(a,DomainStyle(a),target_domain)
CellData.change_domain(a::DistributedCellDatum,input_domain::T,target_domain::T) where T<: DomainStyle = a
CellData.change_domain(a::DistributedCellDatum,input_domain::DomainStyle,target_domain::DomainStyle) = @abstractmethod
CellData.get_array(a::DistributedCellDatum) = get_data(change_domain(a,PhysicalDomain()))

# DistributedCellPoint

struct DistributedCellPoint{A,B} <: DistributedCellDatum
  points::A
  trian::B
  function DistributedCellPoint(
    points::AbstractPData{<:CellPoint},
    trian::DistributedTriangulation)

    A = typeof(points)
    B = typeof(trian)
    new{A,B}(points,trian)
  end
end

CellData.get_data(a::DistributedCellPoint) = map_parts(get_data,a.points)
function CellData.DomainStyle(::Type{<:DistributedCellPoint{A}}) where A
  DomainStyle(eltype(A))
end
CellData.get_triangulation(a::DistributedCellPoint) = a.trian

function CellData.change_domain(
  a::DistributedCellPoint,input_domain::DomainStyle,target_domain::DomainStyle)
  points = map_parts(a.points) do cellpoint
    change_domain(cellpoint,input_domain,target_domain)
  end
  DistributedCellPoint(points,a.trian)
end

# DistributedCellField

struct DistributedCellField{A,B,C} <: DistributedCellDatum
  fields::A
  trian::B
  metadata::C
  function DistributedCellField(
    points::AbstractPData{<:CellField},
    trian::DistributedTriangulation,
    metadata=nothing)

    A = typeof(points)
    B = typeof(trian)
    C = typeof(metadata)
    new{A,B,C}(points,trian)
  end
end

CellData.get_data(a::DistributedCellField) = map_parts(get_data,a.points)
function CellData.DomainStyle(::Type{<:DistributedCellField{A}}) where A
  DomainStyle(eltype(A))
end
CellData.get_triangulation(a::DistributedCellField) = a.trian

function CellData.change_domain(
  a::DistributedCellField,input_domain::DomainStyle,target_domain::DomainStyle)
  points = map_parts(a.points) do cellpoint
    change_domain(cellpoint,input_domain,target_domain)
  end
  DistributedCellField(points,a.trian)
end

# Constructors

function CellData.CellField(f::Function,trian::DistributedTriangulation)
  fields = map_parts(trian.trians) do t
    CellField(f,t)
  end
  DistributedCellField(fields,trian)
end

function CellData.CellField(f::Number,trian::DistributedTriangulation)
  fields = map_parts(trian.trians) do t
    CellField(f,t)
  end
  DistributedCellField(fields,trian)
end

function CellData.CellField(f::AbstractArray{<:Number},trian::DistributedTriangulation)
  fields = map_parts(trian.trians) do t
    CellField(f,t)
  end
  DistributedCellField(fields,trian)
end

# Evaluation

function Arrays.evaluate!(cache,f::DistributedCellField,x::DistributedCellPoint)
  map_parts(f.fields,x.points) do f,x
    f(x)
  end
end

# Operations
