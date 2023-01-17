using Pkg
using PackageCompiler

source_code="""
module sysimagegenerator
   using TestApp
   using PartitionedArrays
   const PArrays = PartitionedArrays
   using MPI

   include("../../mpi/runtests_np4_body.jl")

   with_backend(all_tests,MPIBackend(),(1,1))

   MPI.Finalize()

end #module
"""

warmup_file = joinpath(@__DIR__,"sysimagegenerator.jl")
open(warmup_file,"w") do io
  println(io,source_code)
end

pkgs = Symbol[]
push!(pkgs, :TestApp)

if VERSION >= v"1.4"
    append!(pkgs, [Symbol(v.name) for v in values(Pkg.dependencies()) if v.is_direct_dep],)
else
    append!(pkgs, [Symbol(name) for name in keys(Pkg.installed())])
end

create_sysimage(pkgs,
  sysimage_path=joinpath(@__DIR__,"TestApp.so"),
  precompile_execution_file=warmup_file)
