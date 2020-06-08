
mutable struct MPIPETScCommunicator <: CollaborativeCommunicator
  comm::MPI.Comm
  master_rank::Int
  function MPIPETScCommunicator(comm::MPI.Comm, master_rank::Int = 0)
    comm = new(comm, master_rank)
    finalizer(MPIPETScCommunicatorDestroy,comm)
    comm
  end
end

function MPIPETScCommunicator()
  petsc_comm = Ref{MPI.Comm}(MPI.Comm())
  first_tag  = Ref{PETSc.C.PetscMPIInt}()
  PETSc.C.chk(PETSc.C.PetscCommDuplicate(Float64,MPI.COMM_WORLD,petsc_comm,first_tag))
  MPIPETScCommunicator(petsc_comm[])
end

function MPIPETScCommunicatorDestroy(comm::MPIPETScCommunicator)
  PETSc.PetscFinalized(Float64) || PETSc.C.chk(PETSc.C.PetscCommDuplicate(Float64,comm.comm))
end


# All objects to be used with this communicator need to implement this
# function
function get_part(comm::MPIPETScCommunicator,object,part::Integer)
  @abstractmethod
end

function get_part(comm::MPIPETScCommunicator,object::Number,part::Integer)
  object
end

function i_am_master(comm::MPIPETScCommunicator)
  MPI.Comm_rank(comm.comm) == comm.master_rank
end

function do_on_parts(task::Function, comm::MPIPETScCommunicator, args...)
  part = MPI.Comm_rank(comm.comm) + 1
  largs = map(a->get_part(comm,get_distributed_data(a),part), args)
  task(part, largs...)
end

function num_parts(comm::MPIPETScCommunicator)
  MPI.Comm_size(comm.comm)
end

function num_workers(comm::MPIPETScCommunicator)
  MPI.Comm_size(comm.comm)
end

# We need to compare communicators to perform some checks
function Base.:(==)(a::MPIPETScCommunicator,b::MPIPETScCommunicator)
  @notimplemented
end
