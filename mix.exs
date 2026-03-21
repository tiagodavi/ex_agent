defmodule ExAgent.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_agent,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
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
end
