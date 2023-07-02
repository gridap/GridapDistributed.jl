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

include("Interface.jl")

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

# TO-DO: propose this as a modification to PArrays through PR
function PArrays.with_debug(f,args...;kwargs...)
    f(DebugArray,args...;kwargs...)
end

# TO-DO: propose this as a modification to PArrays through PR
function PArrays.with_mpi(f,args...;comm=MPI.COMM_WORLD,kwargs...)
    if !MPI.Initialized()
        MPI.Init()
    end
    distribute = a -> distribute_with_mpi(a,comm;kwargs...)
    if MPI.Comm_size(comm) == 1
        f(distribute,args...)
    else
        try
            f(distribute,args...)
        catch e
            @error "" exception=(e, catch_backtrace())
            if MPI.Initialized() && !MPI.Finalized()
                MPI.Abort(MPI.COMM_WORLD,1)
            end
        end
    end
    # We are NOT invoking MPI.Finalize() here because we rely on
    # MPI.jl, which registers MPI.Finalize() in atexit()
end

end # module
