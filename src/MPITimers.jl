using Printf

abstract type MPITimerMode end
struct MPITimerModeSum  <: MPITimerMode end
struct MPITimerModeMin  <: MPITimerMode end
struct MPITimerModeLast <: MPITimerMode end

mutable struct MPITimer{M<:MPITimerMode}
  comm    :: MPI.Comm
  message :: String     # Concept being measured (e.g., assembly)
  t_start :: Float64    # last call to start
  t_stop  :: Float64    # last call to stop
  t_accum :: Float64    # result of processing all stop-start
  function MPITimer{M}(comm::MPI.Comm, message) where M<:Union{MPITimerModeSum,MPITimerModeLast}
     new{M}(comm, message, 0.0, 0.0, 0.0)
  end
  function MPITimer{M}(comm::MPI.Comm, message) where M<:MPITimerModeMin
     new{M}(comm, message, 0.0, 0.0, 1.79769e+308)
  end
end

function timer_init(t::MPITimer{M}) where M
   t.t_start=0.0
   t.t_stop =0.0
   if M <: MPITimerModeMin
     t.t_accum   = 1.79769e+308_rp
   end
end

function timer_start(t::MPITimer)
  MPI.Barrier(t.comm)
  t.t_start = MPI.Wtime()
end

function timer_stop(t::MPITimer{M}) where M
  t.t_stop = MPI.Wtime()
  if ( t.t_stop - t.t_start >= 0.0)
    cur_time = t.t_stop - t.t_start
  else
    cur_time = 0.0
  end

  if (M <: MPITimerModeMin)
     if (t.t_accum > cur_time) t.t_accum = cur_time end
  elseif (M <: MPITimerModeSum)
     t.t_accum = t.t_accum + cur_time
  elseif (M <: MPITimerModeLast)
     t.t_accum = cur_time
  end
  t.t_start  = 0.0
  t.t_stop   = 0.0
end

function timer_report(t::MPITimer, show_header::Bool=true)
  accum_max = t.t_accum
  accum_min = t.t_accum
  accum_sum = t.t_accum
  rank = MPI.Comm_rank(t.comm)
  size = MPI.Comm_size(t.comm)
  if (show_header)
    if (rank==0)
      @printf "%25s   %15s  %15s  %15s\n" "" "Min (secs.)" "Max (secs.)" "Avg (secs.)"
    end
  end
  accum_max_reduced = MPI.Reduce([accum_max], MPI.MAX, 0, t.comm)
  accum_min_reduced = MPI.Reduce([accum_min], MPI.MIN, 0, t.comm)
  accum_sum_reduced = MPI.Reduce([accum_sum], +, 0, t.comm)
  if (rank==0)
    @printf "%25s   %15.9e  %15.9e  %15.9e\n"  t.message accum_min_reduced[1] accum_max_reduced[1] accum_sum_reduced[1]/size
  end
  (accum_min_reduced[1],accum_max_reduced[1],accum_sum_reduced[1]/size)
end
