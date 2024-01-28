include("../issue_142.jl")
with_debug() do distribute
    main(distribute,(2,1))
end
