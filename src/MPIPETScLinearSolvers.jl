mutable struct PETScLinearSolver{T} <: LinearSolver
    ksp :: PETSc.KSP{T}
    kws
    function PETScLinearSolver{T}(::Type{T}; kws...) where {T}
        ksp=KSP(PETSc.C.KSP{T}(C_NULL))
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
        mat::PETSc.PetscMat{T}) where {T}
    return PETScSymbolicSetup{T}(ps)
end

function Gridap.Algebra.numerical_setup(
        pss::PETScSymbolicSetup{T},
        mat::PETSc.PetscMat{T}) where {T}
    pss.solver.ksp = KSP(mat; pss.solver.kws...)
    PETSc.KSPSetUp!(pss.solver.ksp)
    return PETScNumericalSetup{T}(pss.solver)
end

function Gridap.Algebra.numerical_setup!(
        pns::PETScNumericalSetup{T},
        mat::PETSc.PetscMat{T}) where {T}
    PETSc.C.chk(PETSc.C.KSPSetOperators(pns.solver.ksp.p,mat.p,mat.p))
    PETSc.KSPSetUp!(pns.solver.ksp)
end

function Gridap.Algebra.solve!(
        x::PETSc.Vec{T},
        ns::PETScNumericalSetup{T},
        b::PETSc.Vec{T}) where {T}
    PETSc.ldiv!(ns.solver.ksp, b, x)
end
