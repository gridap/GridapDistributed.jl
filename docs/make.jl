using Documenter, GridapDistributed

makedocs(;
    modules=[GridapDistributed],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/gridap/GridapDistributed.jl/blob/{commit}{path}#L{line}",
    sitename="GridapDistributed.jl",
    authors="S. Badia <santiago.badia@monash.edu>", "A. F. Martin <alberto.martin@monash.edu>", "F. Verdugo <fverdugo@cimne.upc.edu>",
    assets=String[],
)

deploydocs(;
    repo="github.com/gridap/GridapDistributed.jl",
)
