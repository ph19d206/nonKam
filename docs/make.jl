using Documenter
using nonKam

makedocs(
    sitename = "nonKam.jl",
    modules  = [nonKam],
    remotes  = nothing,
    warnonly = [:missing_docs], # Converts the error to a non-fatal warning
    pages = [
        "Home" => "index.md",
        "Theory" => [
            "Gauge Transformation" => "theory/gauge_transform.md"
        ]
    ]
)