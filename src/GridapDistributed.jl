module GridapDistributed

using Gridap
using Gridap.Helpers
using Gridap.Adaptivity
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.CellData
using Gridap.Visualization
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools

using MPI
using PartitionedArrays
const PArrays = PartitionedArrays

using SparseArrays
using WriteVTK
using FillArrays

import Gridap.TensorValues: inner, outer, double_contraction, symmetric_part
import LinearAlgebra: det, tr, cross, dot, ⋅
import Base: inv, abs, abs2, *, +, -, /, adjoint, transpose, real, imag, conj, getproperty, propertynames
import Gridap.Fields: grad2curl
import Gridap.ODEs.ODETools: ∂t, ∂tt

export FullyAssembledRows
export SubAssembledRows

export get_cell_gids
export get_face_gids

export local_views, get_parts
export with_ghost, no_ghost

include("Algebra.jl")

include("Geometry.jl")

include("CellData.jl")

include("Visualization.jl")

include("FESpaces.jl")

include("DivConformingFESpaces.jl")

include("MultiField.jl")

include("TransientDistributedCellField.jl")

include("TransientMultiFieldDistributedCellField.jl")

include("TransientFESpaces.jl")

include("Adaptivity.jl")

end # module
