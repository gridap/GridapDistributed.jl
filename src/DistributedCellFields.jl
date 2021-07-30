abstract type DistributedCellField <: CellField end;

# Define some of the well known arithmetic ops

#TO-DO: get the list of operators, e.g., from a Gridap constant
for op in (:symmetric_part,:inv,:det,:abs,:abs2,:+,:-,:tr,:transpose,:adjoint,:grad2curl,:real,:imag,:conj)
    @eval begin
      function ($op)(a::DistributedCellField)
        DistributedData(a) do part, a
           Operation($op)(a)
        end
      end
    end
end

# Binary ops
for T in (DistributedCellField,DistributedData{<:CellField},DistributedFEFunction)
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

function (a::typeof(gradient))(x::DistributedData{<:CellField})
  DistributedData(x) do part, x
    a(x)
  end
end
