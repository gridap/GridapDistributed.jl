module GridapDistributed

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.CellData
using Gridap.Visualization
using Gridap.FESpaces
using Gridap.MultiField
using GridapODEs.TransientFETools

using PartitionedArrays
const PArrays = PartitionedArrays

using SparseArrays
using WriteVTK
using FillArrays

import Gridap.TensorValues: inner, outer, double_contraction, symmetric_part
import LinearAlgebra: det, tr, cross, dot, ⋅
import Base: inv, abs, abs2, *, +, -, /, adjoint, transpose, real, imag, conj
import Gridap.Fields: grad2curl
import GridapODEs.ODETools: ∂t, ∂tt

export FullyAssembledRows
export SubAssembledRows

include("Algebra.jl")

include("Geometry.jl")

include("CellData.jl")

include("Visualization.jl")

include("FESpaces.jl")

include("DivConformingFESpaces.jl")

include("MultiField.jl")

include("TransientDistributedCellField.jl")

end # module
