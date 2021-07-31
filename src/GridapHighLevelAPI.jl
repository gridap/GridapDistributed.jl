
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

# Composition (this replaces the @law macro)
for T in (DistributedData{<:CellField},DistributedFEFunction)
  @eval begin
    function Base.:(∘)(f::Function,g::$T)
      DistributedData(g) do part, g
        Operation(f)(g)
      end
    end
    function Base.:(∘)(f::Function,g::Tuple{$T,$T})
      DistributedData(g...) do part, g...
        Operation(f)(g...)
      end
     end
    #  function Base.:(∘)(f::Function,g::Tuple{Vararg{Union{AbstractArray{<:Number},$T}}})
    #     Operation(f)(g...)
    #  end
    #  function Base.:(∘)(f::Function,g::Tuple{Vararg{Union{Function,$T}}})
    #     Operation(f)(g...)
    #  end
  end
end

# Define some of the well known arithmetic ops

#TO-DO: get the list of operators, e.g., from a Gridap constant
for T in (DistributedData{<:CellField},DistributedFEFunction)
  for op in (:symmetric_part,:inv,:det,:abs,:abs2,:+,:-,:tr,:transpose,:adjoint,:grad2curl,:real,:imag,:conj)
     @eval begin
       function ($op)(a::$T)
         DistributedData(a) do part, a
            Operation($op)(a)
         end
       end
     end
  end
 end

 # Binary ops
 for T in (DistributedData{<:CellField},DistributedFEFunction)
  for op in (:inner,:outer,:double_contraction,:+,:-,:*,:cross,:dot,:/)
   @eval begin
     function ($op)(a::$T,b::$T)
       DistributedData(a,b) do part, a, b
         Operation($op)(a,b)
       end
     end
     function ($op)(a::$T,b::Number)
       DistributedData(a) do part, a
         Operation($op)(a,b)
       end
     end
     function ($op)(a::Number,b::$T)
       DistributedData(b) do part, b
         Operation($op)(a,b)
       end
     end
     function ($op)(a::$T,b::Function)
       DistributedData(a) do part, a
         Operation($op)(a,b)
       end
     end
     function ($op)(a::Function,b::$T)
       DistributedData(b) do part, b
         Operation($op)(a,b)
       end
     end
     function ($op)(a::$T,b::AbstractArray{<:Number})
       DistributedData(a) do part, a
         Operation($op)(a,b)
       end
     end
     function ($op)(a::AbstractArray{<:Number},b::$T)
       DistributedData(b) do part, b
         Operation($op)(a,b)
       end
     end
   end
  end
 end

function (+)(a::DistributedData{<:Gridap.CellData.DomainContribution},
             b::DistributedData{<:Gridap.CellData.DomainContribution})
   DistributedData(a,b) do part, a, b
     a+b
   end
end

function (-)(a::DistributedData{<:Gridap.CellData.DomainContribution},
             b::DistributedData{<:Gridap.CellData.DomainContribution})
   DistributedData(a,b) do part, a, b
     a-b
   end
end

function (*)(a::Number,b::DistributedData{<:Gridap.CellData.DomainContribution})
  DistributedData(b) do part, b
    a*b
  end
end

(*)(a::DistributedData{<:Gridap.CellData.DomainContribution},b::Number) = b*a

 function (a::typeof(gradient))(x::DistributedData{<:CellField})
   DistributedData(x) do part, x
     a(x)
   end
 end

 function (a::typeof(gradient))(x::DistributedFEFunction)
  DistributedData(x) do part, x
    a(x)
  end
end

function dot(::typeof(∇),f::DistributedData{<:CellField})
  DistributedData(f) do part, f
    divergence(f)
  end
end


function Base.iterate(a::DistributedData{<:MultiFieldCellField})
  if _num_fields(a)==0
    return nothing
  end
  sf=_get_field(a,1)
  (sf,2)
end

function Base.iterate(a::DistributedData{<:MultiFieldCellField},state)
  if state > _num_fields(a)
     return nothing
  end
  sf=_get_field(a,state)
  (sf,state+1)
end

function _num_fields(a::DistributedData{<:MultiFieldCellField})
  num_fields=0
  do_on_parts(a) do part, a
    num_fields=length(a.single_fields)
  end
  num_fields
end

function _get_field(a::DistributedData{<:MultiFieldCellField},field_id)
  DistributedData(a) do part, a
     a.single_fields[field_id]
  end
end
