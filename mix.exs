defmodule ExAgent.MixProject do
  use Mix.Project

  @version "0.1.0"
  @url_docs "https://hexdocs.pm/ex_agent"
  @url_github "https://github.com/tiagodavi/ex_agent"

  defp description do
    "An Elixir library for building multi-agent LLM applications."
  end

  defp package() do
    [
      files: ~w(lib .formatter.exs mix.exs README* LICENSE* CHANGELOG*),
      licenses: ["Apache-2.0"],
      maintainers: [
        "Tiago D S Batista"
      ],
      links: %{
        "Docs" => @url_docs,
        "Github" => @url_github
      }
    ]
  end

  defp docs do
    [
      source_ref: "v#{@version}",
      source_url: @url_github,
      main: "ExAgent",
      extra_section: "guides",
      extras: ["README.md", "NOTICE", "LICENSE"]
    ]
  end

  def project do
    [
      app: :ex_agent,
      name: "ExAgent",
      source_url: @url_github,
      homepage_url: @url_docs,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      deps: deps(),
      docs: docs(),
      elixirc_paths: elixirc_paths(Mix.env()),
      elixirc_options: [ignore_module_conflict: true],
      source_url: @url_github
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {ExAgent.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:req, "~> 0.5"},
      {:jason, "~> 1.4"},
      {:nimble_options, "~> 1.1"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:mox, "~> 1.1", only: :test},
      {:plug, "~> 1.16", only: :test}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]
end
