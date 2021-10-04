module MPITimersTests
  using MPI
  using GridapDistributed
  if (!MPI.Initialized())
     MPI.Init()
  end
  timer=GridapDistributed.MPITimer{GridapDistributed.MPITimerModeMin}(MPI.COMM_WORLD,"sleep(rand())")
  GridapDistributed.timer_start(timer)
  sleep(rand())
  GridapDistributed.timer_stop(timer)
  GridapDistributed.timer_start(timer)
  sleep(rand())
  GridapDistributed.timer_stop(timer)
  GridapDistributed.timer_report(timer)
end
