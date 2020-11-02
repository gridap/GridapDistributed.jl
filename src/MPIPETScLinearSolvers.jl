mutable struct PETScLinearSolver{T} <: LinearSolver
    ksp :: GridapDistributedPETScWrappers.KSP{T}
    kws
    function PETScLinearSolver{T}(::Type{T}; kws...) where {T}
        ksp=KSP(GridapDistributedPETScWrappers.C.KSP{T}(C_NULL))
        new{T}(ksp,kws)
    end
end

PETScLinearSolver(::Type{T}; kws...) where {T}=PETScLinearSolver{T}(T;kws...)

struct PETScSymbolicSetup{T} <: SymbolicSetup
    solver :: PETScLinearSolver{T}
end

struct PETScNumericalSetup{T} <: NumericalSetup
    solver :: PETScLinearSolver{T}
end

function Gridap.Algebra.symbolic_setup(
        ps::PETScLinearSolver{T},
        mat::GridapDistributedPETScWrappers.PetscMat{T}) where {T}
    return PETScSymbolicSetup{T}(ps)
end

function Gridap.Algebra.numerical_setup(
        pss::PETScSymbolicSetup{T},
        mat::GridapDistributedPETScWrappers.PetscMat{T}) where {T}
    pss.solver.ksp = KSP(mat; pss.solver.kws...)
    GridapDistributedPETScWrappers.KSPSetUp!(pss.solver.ksp)
    return PETScNumericalSetup{T}(pss.solver)
end

function Gridap.Algebra.numerical_setup!(
        pns::PETScNumericalSetup{T},
        mat::GridapDistributedPETScWrappers.PetscMat{T}) where {T}
    GridapDistributedPETScWrappers.C.chk(GridapDistributedPETScWrappers.C.KSPSetOperators(pns.solver.ksp.p,mat.p,mat.p))
    GridapDistributedPETScWrappers.KSPSetUp!(pns.solver.ksp)
end

function Gridap.Algebra.solve!(
        x::GridapDistributedPETScWrappers.Vec{T},
        ns::PETScNumericalSetup{T},
        b::GridapDistributedPETScWrappers.Vec{T}) where {T}
    GridapDistributedPETScWrappers.ldiv!(ns.solver.ksp, b, x)
end
