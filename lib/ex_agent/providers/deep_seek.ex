defmodule ExAgent.Providers.DeepSeek do
  @moduledoc """
  DeepSeek LLM provider.

  DeepSeek uses an OpenAI-compatible API format with Bearer token auth.

  ## Example

      provider = ExAgent.Providers.DeepSeek.new(api_key: "sk-...")
      ExAgent.LlmProvider.chat(provider, messages)
  """

  @type t :: %__MODULE__{
          api_key: String.t(),
          model: String.t(),
          base_url: String.t(),
          temperature: float(),
          max_tokens: integer(),
          system_prompt: String.t() | nil,
          tools: [ExAgent.Tool.t()],
          req: Req.Request.t() | nil
        }

  @enforce_keys [:api_key]
  defstruct [
    :api_key,
    :system_prompt,
    :req,
    model: "deepseek-chat",
    base_url: "https://api.deepseek.com/v1",
    temperature: 0.6,
    max_tokens: 512,
    tools: []
  ]

  @schema [
    api_key: [type: :string, required: true, doc: "DeepSeek API key"],
    model: [type: :string, default: "deepseek-chat", doc: "Model name"],
    base_url: [type: :string, default: "https://api.deepseek.com/v1", doc: "API base URL"],
    temperature: [type: :float, default: 0.6],
    max_tokens: [type: :pos_integer, default: 512],
    system_prompt: [type: {:or, [:string, nil]}, default: nil, doc: "System prompt"],
    tools: [type: {:list, :any}, default: [], doc: "Available tools"]
  ]

  @doc """
  Creates a new DeepSeek provider with validated options and initialized Req client.

  ## Options

  - `:api_key` (required) - DeepSeek API key
  - `:model` - Model name (default: `"deepseek-chat"`)
  - `:base_url` - API base URL (default: `"https://api.deepseek.com/v1"`)
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
      ExAgent.Services.DeepSeekService.chat(provider, messages, opts)
    end
  end
end
