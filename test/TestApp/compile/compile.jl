function generate_precompile_execution_file()
 source_code="""
 module sysimagegenerator
    using TestApp
    using PartitionedArrays
    const PArrays = PartitionedArrays
    using MPI

    if ! MPI.Initialized()
      MPI.Init()
    end

    parts = get_part_ids(mpi,(1,1))

    include("../../mpi/runtests_np4_body.jl")

    MPI.Finalize()

 end #module
 """
 open("sysimagegenerator.jl","w") do io
   println(io,source_code)
 end
end

using Pkg
using PackageCompiler
generate_precompile_execution_file()
pkgs = Symbol[]
push!(pkgs, :TestApp)

if VERSION >= v"1.4"
    append!(pkgs, [Symbol(v.name) for v in values(Pkg.dependencies()) if v.is_direct_dep],)
else
    append!(pkgs, [Symbol(name) for name in keys(Pkg.installed())])
end

create_sysimage(pkgs,
  sysimage_path=joinpath(@__DIR__,"TestApp.so"),
  precompile_execution_file=joinpath(@__DIR__,"sysimagegenerator.jl"))
