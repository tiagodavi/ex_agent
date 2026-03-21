defmodule ExAgent.Providers.OpenAI do
  @moduledoc """
  OpenAI LLM provider.

  Wraps the OpenAI chat completions API with automatic Req client
  configuration, Bearer token auth, and JSON encoding.

  ## Example

      provider = ExAgent.Providers.OpenAI.new(api_key: "sk-...")
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
    model: "gpt-4o",
    base_url: "https://api.openai.com/v1",
    tools: []
  ]

  @schema [
    api_key: [type: :string, required: true, doc: "OpenAI API key"],
    model: [type: :string, default: "gpt-4o", doc: "Model name"],
    base_url: [type: :string, default: "https://api.openai.com/v1", doc: "API base URL"],
    system_prompt: [type: {:or, [:string, nil]}, default: nil, doc: "System prompt"],
    tools: [type: {:list, :any}, default: [], doc: "Available tools"]
  ]

  @doc """
  Creates a new OpenAI provider with validated options and initialized Req client.

  ## Options

  - `:api_key` (required) - OpenAI API key
  - `:model` - Model name (default: `"gpt-4o"`)
  - `:base_url` - API base URL (default: `"https://api.openai.com/v1"`)
  - `:system_prompt` - System prompt to prepend to messages
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
      headers: [{"authorization", "Bearer #{key}"}]
    )
  end

  defimpl ExAgent.LlmProvider do
    def chat(provider, messages, opts \\ []) do
      ExAgent.Services.OpenAIService.chat(
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
