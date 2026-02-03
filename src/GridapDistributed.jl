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
using Gridap.ODEs

using MPI
using PartitionedArrays
const PArrays = PartitionedArrays
using PartitionedArrays: getany

using SparseArrays
using WriteVTK
using FillArrays
using BlockArrays
using LinearAlgebra
using ForwardDiff
using CircularArrays

import Gridap.TensorValues: inner, outer, double_contraction, symmetric_part
import LinearAlgebra: det, tr, cross, dot, ⋅, diag
import Base: inv, abs, abs2, *, +, -, /, adjoint, transpose, real, imag, conj, getproperty, propertynames
import Gridap.Fields: grad2curl
import Gridap.CellData: Interpolable

export FullyAssembledRows
export SubAssembledRows

export get_cell_gids
export get_face_gids

export local_views, get_parts
export with_ghost, no_ghost

export redistribute

include("PArraysExtras.jl")

include("BlockPartitionedArrays.jl")

include("Algebra.jl")

include("Geometry.jl")

include("CellData.jl")

include("Visualization.jl")

include("FESpaces.jl")

# include("DivConformingFESpaces.jl")

include("MultiField.jl")

include("ODEs.jl")

include("Redistribution.jl")

include("Adaptivity.jl")

include("Autodiff.jl")

include("MacroDiscreteModels.jl")

include("PatchAssemblers.jl")

end # module
