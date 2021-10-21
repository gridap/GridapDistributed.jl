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

    display(parts)

    t = PArrays.PTimer(parts,verbose=true)
    PArrays.tic!(t)

    TestApp.GeometryTests.main(parts)
    PArrays.toc!(t,"Geometry")

    TestApp.CellDataTests.main(parts)
    PArrays.toc!(t,"CellData")

    TestApp.FESpacesTests.main(parts)
    PArrays.toc!(t,"FESpaces")

    TestApp.MultiFieldTests.main(parts)
    PArrays.toc!(t,"MultiField")

    TestApp.PoissonTests.main(parts)
    PArrays.toc!(t,"Poisson")

    TestApp.PLaplacianTests.main(parts)
    PArrays.toc!(t,"PLaplacian")

    display(t)

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
