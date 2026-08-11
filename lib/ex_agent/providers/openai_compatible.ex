defmodule ExAgent.Providers.OpenAICompatible do
  @moduledoc """
  Provider for any endpoint speaking the OpenAI chat-completions dialect.

  Covers self-hosted vLLM (including behind Modal), OpenRouter, Together, Groq,
  and anything else exposing `POST {base_url}/chat/completions`.

  ## Modal (vLLM behind Modal's proxy auth)

      provider = ExAgent.Providers.OpenAICompatible.new(
        base_url: System.fetch_env!("MODAL_QWEN_URL") <> "/v1",
        model: "Qwen/Qwen3-VL-8B-Instruct",
        headers: [
          {"Modal-Key", System.fetch_env!("MODAL_KEY")},
          {"Modal-Secret", System.fetch_env!("MODAL_SECRET")}
        ],
        modalities: [:text, :image, :video]
      )

      {:ok, agent} = ExAgent.start_agent(provider: provider)

  ## OpenRouter / Together / Groq (bearer auth)

      provider = ExAgent.Providers.OpenAICompatible.new(
        base_url: "https://openrouter.ai/api/v1",
        model: "meta-llama/llama-3.3-70b-instruct",
        api_key: System.fetch_env!("OPENROUTER_API_KEY"),
        modalities: [:text, :image]
      )

  `:api_key` is sugar for an `Authorization: Bearer` header. `:headers` are merged
  afterwards, so an explicit header always wins.

  > #### `:base_url` includes the version prefix {: .info}
  >
  > As with `ExAgent.Providers.OpenAI`, `:base_url` is the full API root — pass
  > `https://host/v1`, not `https://host`. Requests are issued against
  > `\#{base_url}/chat/completions`.

  ## Declaring modalities

  `:modalities` describes *this deployment*, not the module: one vLLM container
  serves one model, and Qwen3-VL handles video where a text-only Llama container
  does not. It defaults to `[:text]`, so a misconfigured deployment fails loudly
  rather than firing video at a model that cannot read it.

  ## No Files API

  Compatible endpoints have no upload endpoint, so bytes are always sent as
  `data:` URIs. Past `:max_inline_bytes` (32 MB by default) the call returns
  `{:error, %ExAgent.Error{type: :unsupported}}` telling you to host the asset at
  a URL the container can reach — it is never silently truncated or retried.

  Documents are not supported: there is no portable content part for them.
  Extract the text or rasterize the pages first.

  ## Verifying the served model

  Container swaps where config still names the old model are a common self-host
  failure. `probe/1` checks:

      case ExAgent.Providers.OpenAICompatible.probe(provider) do
        :ok -> :ready
        {:error, error} -> Logger.warning(Exception.message(error))
      end
  """

  @behaviour ExAgent.Provider

  require Logger

  alias ExAgent.Error

  @type t :: %__MODULE__{
          base_url: String.t(),
          model: String.t(),
          api_key: String.t() | nil,
          headers: [{String.t(), String.t()}],
          modalities: [ExAgent.Source.modality()],
          max_inline_bytes: pos_integer(),
          temperature: float() | nil,
          max_tokens: pos_integer() | nil,
          system_prompt: String.t() | nil,
          tools: [ExAgent.Tool.t()],
          req: Req.Request.t() | nil
        }

  @enforce_keys [:base_url, :model]
  defstruct [
    :base_url,
    :model,
    :api_key,
    :system_prompt,
    :req,
    headers: [],
    modalities: [:text],
    max_inline_bytes: 33_554_432,
    temperature: 0.6,
    max_tokens: 512,
    tools: []
  ]

  @schema [
    base_url: [type: :string, required: true, doc: "API root, including any `/v1` prefix"],
    model: [type: :string, required: true, doc: "Model name as the endpoint serves it"],
    api_key: [
      type: {:or, [:string, nil]},
      default: nil,
      doc: "Sugar for an `Authorization: Bearer` header"
    ],
    headers: [
      type: {:list, :any},
      default: [],
      doc: "Extra request headers, merged after `:api_key` so they win"
    ],
    modalities: [
      type: {:list, :atom},
      default: [:text],
      doc: "Attachment modalities this deployment accepts"
    ],
    max_inline_bytes: [
      type: :pos_integer,
      default: 33_554_432,
      doc: "Largest attachment sent as a data URI; there is no upload fallback"
    ],
    temperature: [type: {:or, [:float, nil]}, default: 0.6],
    max_tokens: [type: {:or, [:pos_integer, nil]}, default: 512],
    system_prompt: [type: {:or, [:string, nil]}, default: nil, doc: "System prompt"],
    tools: [type: {:list, :any}, default: [], doc: "Available tools"]
  ]

  @doc """
  Creates a provider with validated options and an initialized Req client.

  Performs no network I/O — use `probe/1` to verify the served model.

  ## Options

  #{NimbleOptions.docs(@schema)}
  """
  @spec new(keyword()) :: t()
  def new(opts) do
    validated = NimbleOptions.validate!(opts, @schema)
    provider = struct!(__MODULE__, validated)
    %{provider | req: build_req(provider)}
  end

  @doc """
  Checks that the endpoint serves the configured model.

  Issues `GET {base_url}/models` and compares the result against `:model`.
  Returns `:ok` when the model is listed, and an `ExAgent.Error` otherwise —
  including when the endpoint is unreachable.

  A mismatch is reported rather than raised: some gateways do not enumerate
  every model they will route.
  """
  @spec probe(t()) :: :ok | {:error, Error.t()}
  def probe(%__MODULE__{} = provider) do
    # A health check should answer promptly; Req's default 5xx retry would make
    # an unreachable endpoint block for seconds.
    provider.req
    |> Req.get(url: "/models", retry: false)
    |> Error.from_result(__MODULE__)
    |> case do
      {:ok, body} ->
        served = served_models(body)

        if provider.model in served do
          :ok
        else
          {:error,
           Error.new(
             :not_found,
             "#{provider.base_url} does not list model #{inspect(provider.model)}; " <>
               "it serves #{inspect(served)}",
             __MODULE__
           )}
        end

      {:error, _error} = failure ->
        failure
    end
  end

  @impl true
  def chat(provider, messages, opts \\ []) do
    ExAgent.Services.OpenAICompatibleService.chat(provider, messages, opts)
  end

  @impl true
  def stream(provider, messages, opts \\ []) do
    ExAgent.Services.OpenAICompatibleService.stream(provider, messages, opts)
  end

  @impl true
  def supported_modalities(%__MODULE__{modalities: modalities}), do: modalities

  @impl true
  def embed(provider, inputs, opts \\ []) do
    ExAgent.Services.OpenAICompatibleEmbedService.embed(provider, inputs, opts)
  end

  @spec build_req(t()) :: Req.Request.t()
  defp build_req(%__MODULE__{} = provider) do
    Req.new(
      base_url: provider.base_url,
      headers: merge_headers(auth_headers(provider), provider.headers)
    )
  end

  @spec auth_headers(t()) :: [{String.t(), String.t()}]
  defp auth_headers(%__MODULE__{api_key: nil}), do: []
  defp auth_headers(%__MODULE__{api_key: key}), do: [{"authorization", "Bearer #{key}"}]

  # Req appends same-named headers rather than replacing them, which would send
  # two Authorization values. Drop any default an explicit header overrides.
  @spec merge_headers([{String.t(), String.t()}], [{String.t(), String.t()}]) ::
          [{String.t(), String.t()}]
  defp merge_headers(defaults, overrides) do
    overridden = MapSet.new(overrides, fn {name, _value} -> String.downcase(name) end)

    Enum.reject(defaults, fn {name, _value} -> String.downcase(name) in overridden end) ++
      overrides
  end

  @spec served_models(term()) :: [String.t()]
  defp served_models(%{"data" => models}) when is_list(models) do
    Enum.flat_map(models, fn
      %{"id" => id} when is_binary(id) -> [id]
      _ -> []
    end)
  end

  defp served_models(_body), do: []
end
