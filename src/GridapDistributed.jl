module GridapDistributed

using Gridap
using Gridap.Helpers
using Gridap.Arrays
using Gridap.Fields
using Gridap.Geometry
using Gridap.CellData
using Gridap.Visualization

using PartitionedArrays
const PArrays = PartitionedArrays

import Gridap.TensorValues: inner, outer, double_contraction, symmetric_part
import LinearAlgebra: det, tr, cross, dot, ⋅
import Base: inv, abs, abs2, *, +, -, /, adjoint, transpose, real, imag, conj
import Gridap.Fields: grad2curl

include("Geometry.jl")

include("CellData.jl")

include("Visualization.jl")

#include("FESpaces.jl")

end # module
