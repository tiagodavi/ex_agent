defmodule ExAgent.Providers.OpenAI do
  @moduledoc """
  OpenAI LLM provider.

  Wraps the OpenAI chat completions API with automatic Req client
  configuration, Bearer token auth, and JSON encoding.

  ## Example

      provider = ExAgent.Providers.OpenAI.new(api_key: "sk-...")
      ExAgent.Provider.chat(provider, messages)
  """

  @behaviour ExAgent.Provider

  @type t :: %__MODULE__{
          api_key: String.t(),
          model: String.t(),
          base_url: String.t(),
          temperature: float() | nil,
          max_tokens: integer() | nil,
          system_prompt: String.t() | nil,
          tools: [ExAgent.Tool.t()],
          modalities: [ExAgent.Source.modality()],
          upload_cache: boolean(),
          req: Req.Request.t() | nil
        }

  @enforce_keys [:api_key]
  defstruct [
    :api_key,
    :system_prompt,
    :req,
    model: "gpt-4o",
    base_url: "https://api.openai.com/v1",
    temperature: 0.6,
    max_tokens: 512,
    tools: [],
    modalities: [:text, :image, :document],
    upload_cache: true
  ]

  @schema [
    api_key: [type: :string, required: true, doc: "OpenAI API key"],
    model: [type: :string, default: "gpt-4o", doc: "Model name"],
    base_url: [type: :string, default: "https://api.openai.com/v1", doc: "API base URL"],
    temperature: [type: {:or, [:float, nil]}, default: 0.6],
    max_tokens: [type: {:or, [:pos_integer, nil]}, default: 512],
    system_prompt: [type: {:or, [:string, nil]}, default: nil, doc: "System prompt"],
    tools: [type: {:list, :any}, default: [], doc: "Available tools"],
    modalities: [
      type: {:list, :atom},
      default: [:text, :image, :document],
      doc: "Attachment modalities the chosen model accepts"
    ],
    upload_cache: [
      type: :boolean,
      default: true,
      doc: "Reuse previous uploads of identical bytes via `ExAgent.UploadCache`"
    ]
  ]

  @doc """
  Creates a new OpenAI provider with validated options and initialized Req client.

  ## Options

  - `:api_key` (required) - OpenAI API key
  - `:model` - Model name (default: `"gpt-4o"`)
  - `:base_url` - API base URL (default: `"https://api.openai.com/v1"`)
  - `:system_prompt` - System prompt to prepend to messages
  - `:tools` - List of `ExAgent.Tool` structs
  - `:modalities` - Attachment modalities the chosen model accepts
    (default: `[:text, :image, :document]`). Narrow it for a text-only model such
    as `o1-mini` so an attachment is rejected up front instead of 400ing at the API.
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

  @impl true
  def chat(provider, messages, opts \\ []) do
    ExAgent.Services.OpenAIService.chat(provider, messages, opts)
  end

  @impl true
  def upload(provider, file_data, mime_type, opts \\ []) do
    ExAgent.Services.OpenAIUploadService.upload(provider.req, file_data, mime_type, opts)
  end

  @impl true
  def stream(provider, messages, opts \\ []) do
    ExAgent.Services.OpenAIService.stream(provider, messages, opts)
  end

  @impl true
  def supported_modalities(%__MODULE__{modalities: modalities}), do: modalities

  @impl true
  def embed(provider, inputs, opts \\ []) do
    ExAgent.Services.OpenAIEmbedService.embed(provider, inputs, opts)
  end
end
