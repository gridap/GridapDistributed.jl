module GridapODEs

using GridapDistributed: DistributedCellDatum, DistributedCellField, DistributedMeasure
using GridapDistributed: DistributedSingleFieldFEFunction, DistributedMultiFieldFEFunction
using GridapDistributed: DistributedSingleFieldFESpace, DistributedMultiFieldFESpace
using GridapDistributed: local_views
using Gridap.Fields
using Gridap.Geometry
using Gridap.Arrays
using Gridap.CellData
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField: num_fields
using Gridap.Helpers: first_and_tail
using GridapODEs.TransientFETools
using GridapODEs.ODETools
using PartitionedArrays

import GridapODEs.ODETools: ∂t, ∂tt
import Gridap.TensorValues: inner, outer, double_contraction, symmetric_part
import LinearAlgebra: det, tr, cross, dot, ⋅
import Base: inv, abs, abs2, *, +, -, /, adjoint, transpose, real, imag, conj
import Gridap.Fields: grad2curl

export TransientDistributedCellField
export TransientSingleFieldDistributedCellField
export TransientMultiFieldDistributedCellField

include("TransientDistributedCellField.jl")

include("TransientMultiFieldDistributedCellField.jl")

include("TransientFESpaces.jl")

end
