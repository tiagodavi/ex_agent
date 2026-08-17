# Used by "mix format"
[
  # Ecto's schema DSL is formatted without parens, matching how users write the
  # embedded schemas passed to `:schema`.
  import_deps: [:ecto],
  inputs: ["{mix,.formatter}.exs", "{config,lib,test}/**/*.{ex,exs}"]
]
