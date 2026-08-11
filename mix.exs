defmodule ExAgent.MixProject do
  use Mix.Project

  @version "0.3.0"
  @url_docs "https://hexdocs.pm/ex_agent"
  @url_github "https://github.com/tiagodavi/ex_agent"

  defp description do
    "An Elixir library for building multi-agent LLM applications."
  end

  defp package() do
    [
      files: ~w(lib assets .formatter.exs mix.exs README* LICENSE* CHANGELOG* NOTICE),
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
      main: "readme",
      extra_section: "guides",
      extras: ["README.md", "CHANGELOG.md", "NOTICE", "LICENSE"],
      # Copied verbatim so the README's `assets/` image paths resolve the same
      # way on HexDocs as they do on GitHub.
      assets: %{"assets" => "assets"},
      # Grouped to mirror the README's sections, so the sidebar reads the same
      # way as the guide.
      groups_for_modules: [
        "Core API": [ExAgent],
        "Providers & roles": [
          ExAgent.Provider,
          ExAgent.Providers.OpenAI,
          ExAgent.Providers.Gemini,
          ExAgent.Providers.OpenAICompatible,
          ExAgent.Providers.JinaV5,
          ExAgent.Providers.JinaRerankerM0,
          ExAgent.Roles
        ],
        Conversations: [
          ExAgent.Agent,
          ExAgent.Message,
          ExAgent.Response,
          ExAgent.Chunk,
          ExAgent.Context,
          ExAgent.Tool,
          ExAgent.Skill,
          ExAgent.Error,
          ExAgent.Telemetry
        ],
        Files: [
          ExAgent.Attachment,
          ExAgent.Source,
          ExAgent.FileRef,
          ExAgent.UploadCache
        ],
        Embeddings: [ExAgent.Embeddings],
        Reranking: [ExAgent.Reranking],
        "Multi-agent patterns": [
          ExAgent.Patterns.Chain,
          ExAgent.Patterns.Router,
          ExAgent.Patterns.Subagents,
          ExAgent.Patterns.Handoff,
          ExAgent.Patterns.Skills,
          ExAgent.Patterns.Reflection,
          ExAgent.Patterns.MapReduce,
          ExAgent.Patterns.Consensus
        ],
        Internals: [
          ExAgent.SSE,
          ExAgent.AgentSupervisor,
          ExAgent.AgentDynamicSupervisor,
          ExAgent.Services.OpenAIService,
          ExAgent.Services.OpenAIEmbedService,
          ExAgent.Services.OpenAIUploadService,
          ExAgent.Services.OpenAICompatibleService,
          ExAgent.Services.OpenAICompatibleEmbedService,
          ExAgent.Services.GeminiService,
          ExAgent.Services.GeminiEmbedService,
          ExAgent.Services.GeminiUploadService,
          ExAgent.Services.Streaming
        ]
      ]
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
      {:telemetry, "~> 1.0"},
      {:jason, "~> 1.4"},
      {:mime, "~> 2.0"},
      {:nimble_options, "~> 1.1"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:mox, "~> 1.1", only: :test},
      {:plug, "~> 1.16", only: :test}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]
end
