defmodule ExAgent.Providers.Gemini do
  @moduledoc """
  Google Gemini LLM provider.

  Wraps the Gemini `generateContent` API with automatic Req client
  configuration, API key header, and JSON encoding.

  ## Example

      provider = ExAgent.Providers.Gemini.new(api_key: "AIza...")
      ExAgent.LlmProvider.chat(provider, messages)
  """

  @type t :: %__MODULE__{
          api_key: String.t(),
          model: String.t(),
          base_url: String.t(),
          system_prompt: String.t() | nil,
          tools: [ExAgent.Tool.t()],
          req: Req.Request.t() | nil
        }

  @enforce_keys [:api_key]
  defstruct [
    :api_key,
    :system_prompt,
    :req,
    model: "gemini-2.0-flash",
    base_url: "https://generativelanguage.googleapis.com/v1beta",
    tools: []
  ]

  @schema [
    api_key: [type: :string, required: true, doc: "Google API key"],
    model: [type: :string, default: "gemini-2.0-flash", doc: "Model name"],
    base_url: [
      type: :string,
      default: "https://generativelanguage.googleapis.com/v1beta",
      doc: "API base URL"
    ],
    system_prompt: [type: {:or, [:string, nil]}, default: nil, doc: "System prompt"],
    tools: [type: {:list, :any}, default: [], doc: "Available tools"]
  ]

  @doc """
  Creates a new Gemini provider with validated options and initialized Req client.

  ## Options

  - `:api_key` (required) - Google API key
  - `:model` - Model name (default: `"gemini-2.0-flash"`)
  - `:base_url` - API base URL
  - `:system_prompt` - System instruction to prepend
  - `:tools` - List of `ExAgent.Tool` structs
  """
  @spec new(keyword()) :: t()
  def new(opts) do
    validated = NimbleOptions.validate!(opts, @schema)
    provider = struct!(__MODULE__, validated)
    %{provider | req: build_req(provider)}
  end

  @spec build_req(t()) :: Req.Request.t()
  defp build_req(%__MODULE__{api_key: key, base_url: url}) do
    Req.new(
      base_url: url,
      headers: [{"x-goog-api-key", key}]
    )
  end

  defimpl ExAgent.LlmProvider do
    def chat(provider, messages, opts \\ []) do
      ExAgent.Services.GeminiService.chat(
        provider.req,
        provider.model,
        messages,
        provider.tools,
        provider.system_prompt,
        opts
      )
    end
  end
end
