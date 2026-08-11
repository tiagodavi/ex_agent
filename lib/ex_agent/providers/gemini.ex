defmodule ExAgent.Providers.Gemini do
  @moduledoc """
  Google Gemini LLM provider.

  Wraps the Gemini `generateContent` API with automatic Req client
  configuration, API key header, and JSON encoding.

  ## Example

      provider = ExAgent.Providers.Gemini.new(api_key: "AIza...")
      ExAgent.Provider.chat(provider, messages)
  """

  @behaviour ExAgent.Provider

  # See `ExAgent.Providers.OpenAI` — the credential must not reach a crash dump.
  @derive {Inspect, except: [:api_key, :req]}

  @type t :: %__MODULE__{
          api_key: String.t(),
          model: String.t(),
          base_url: String.t(),
          temperature: float() | nil,
          max_tokens: pos_integer() | nil,
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
    model: "gemini-3.6-flash",
    base_url: "https://generativelanguage.googleapis.com/v1beta",
    temperature: nil,
    max_tokens: nil,
    tools: [],
    modalities: [:text, :image, :document, :video, :audio],
    upload_cache: true
  ]

  @schema [
    api_key: [type: :string, required: true, doc: "Google API key"],
    model: [type: :string, default: "gemini-3.6-flash", doc: "Model name"],
    base_url: [
      type: :string,
      default: "https://generativelanguage.googleapis.com/v1beta",
      doc: "API base URL"
    ],
    temperature: [
      type: {:or, [:float, :integer, nil]},
      default: nil,
      doc: "Sampling temperature; omitted from the request when nil"
    ],
    max_tokens: [
      type: {:or, [:pos_integer, nil]},
      default: nil,
      doc: "Sent as `maxOutputTokens`; omitted when nil"
    ],
    system_prompt: [type: {:or, [:string, nil]}, default: nil, doc: "System prompt"],
    tools: [type: {:list, :any}, default: [], doc: "Available tools"],
    modalities: [
      type: {:list, :atom},
      default: [:text, :image, :document, :video, :audio],
      doc: "Attachment modalities the chosen model accepts"
    ],
    upload_cache: [
      type: :boolean,
      default: true,
      doc: "Reuse previous uploads of identical bytes via `ExAgent.UploadCache`"
    ]
  ]

  @doc """
  Creates a new Gemini provider with validated options and initialized Req client.

  ## Options

  - `:api_key` (required) - Google API key
  - `:model` - Model name (default: `"gemini-3.6-flash"`)
  - `:base_url` - API base URL
  - `:system_prompt` - System instruction to prepend
  - `:tools` - List of `ExAgent.Tool` structs
  - `:upload_cache` - Reuse previous uploads of identical bytes (default: `true`)
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

  @impl true
  def chat(provider, messages, opts \\ []) do
    ExAgent.Services.GeminiService.chat(provider, messages, opts)
  end

  @impl true
  def upload(provider, file_data, mime_type, opts \\ []) do
    ExAgent.Services.GeminiUploadService.upload(provider.api_key, file_data, mime_type, opts)
  end

  @impl true
  def stream(provider, messages, opts \\ []) do
    ExAgent.Services.GeminiService.stream(provider, messages, opts)
  end

  @impl true
  def supported_modalities(%__MODULE__{modalities: modalities}), do: modalities

  @impl true
  def embed(provider, inputs, opts \\ []) do
    ExAgent.Services.GeminiEmbedService.embed(provider, inputs, opts)
  end

  @impl true
  def embedding_tasks(_provider), do: ExAgent.Services.GeminiEmbedService.tasks()
end
