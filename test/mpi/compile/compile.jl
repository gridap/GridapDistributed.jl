using Pkg
Pkg.add("PackageCompiler")
using PackageCompiler

@assert length(ARGS)==1
test_script=ARGS[1]
@assert isfile("../../"*test_script)
modulename=split(test_script,".")[1]

function generate_precompile_execution_file(modulename)
 source_code="""
 module sysimagegenerator 

 using PartitionedArrays
 const PArrays = PartitionedArrays
 using MPI

 if ! MPI.Initialized()
   MPI.Init()
 end

 include(\"../../$(modulename).jl\")

 @assert MPI.Comm_size(MPI.COMM_WORLD) == 1
 parts = get_part_ids(mpi,(1,1))

 display(parts)

 t = PArrays.PTimer(parts,verbose=true)
 PArrays.tic!(t)

 eval(Meta.parse(\"$(modulename).main(parts)\"))
 PArrays.toc!(t,\"$(modulename)\")
 display(t)

 MPI.Finalize()

 end #module
 """
 open("sysimagegenerator.jl","w") do io
   println(io,source_code)
 end
end 

@assert length(ARGS)==1
test_script=ARGS[1]
@assert isfile("../../"*test_script)
modulename=split(test_script,".")[1]
generate_precompile_execution_file(modulename)

pkgs = Symbol[]
push!(pkgs, :GridapDistributed)

if VERSION >= v"1.4"
    append!(pkgs, [Symbol(v.name) for v in values(Pkg.dependencies()) if v.is_direct_dep],)
else
    append!(pkgs, [Symbol(name) for name in keys(Pkg.installed())])
end

create_sysimage(pkgs,
  sysimage_path=joinpath(@__DIR__,"GridapDistributedMPIBackend.so"),
  precompile_execution_file=joinpath(@__DIR__,"sysimagegenerator.jl"))
