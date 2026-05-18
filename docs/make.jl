using Documenter, GridapDistributed

pages = [
  "Home"          => "index.md",
  "Backends"      => "backends.md",
  "Algebra"       => "algebra.md",
  "Geometry"      => "geometry.md",
  "FESpaces"      => "fespaces.md",
  "Assembly"      => "assembly.md",
  "Adaptivity"    => "adaptivity.md",
  "Visualization" => "visualization.md",
]

makedocs(;
    modules=[GridapDistributed],
    format=Documenter.HTML(),
    pages=pages,
    repo="https://github.com/gridap/GridapDistributed.jl/blob/{commit}{path}#L{line}",
    sitename="GridapDistributed.jl",
    authors="S. Badia <santiago.badia@monash.edu>, A. F. Martin <alberto.f.martin@anu.edu.au>, F. Verdugo <fverdugo@cimne.upc.edu>",
    warnonly=true,
)

deploydocs(;
    repo="github.com/gridap/GridapDistributed.jl",
)
