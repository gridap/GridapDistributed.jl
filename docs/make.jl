using Documenter, GridapDistributed

pages = [
  "Home" => "index.md",
  "Getting Started" => "getting-started.md",
  "GridapDistributed" => "GridapDistributed.md",
  "Algebra" => "Algebra.md",
  "CellData" => "CellData.md",
  "DivConformingFESpaces" => "DivConformingFESpaces.md",
  "FESpaces" => "FESpaces.md",
  "Geometry" => "Geometry.md",
  "MultiField" => "MultiField.md",
  "Visualization" => "Visualization.md",
  "Adaptivity" => "Adaptivity.md",
 ]

makedocs(;
    modules=[GridapDistributed],
    format=Documenter.HTML(),
    pages=pages,
    repo="https://github.com/gridap/GridapDistributed.jl/blob/{commit}{path}#L{line}",
    sitename="GridapDistributed.jl",
    authors="S. Badia <santiago.badia@monash.edu>, A. F. Martin <alberto.f.martin@anu.edu.au>, F. Verdugo <fverdugo@cimne.upc.edu>",
)

deploydocs(;
    repo="github.com/gridap/GridapDistributed.jl",
)
