defmodule ExAgent.Providers.JinaV5 do
  @moduledoc """
  Jina embeddings v5 — an **embeddings-only** provider for a self-hosted server.

      provider = ExAgent.Providers.JinaV5.new(base_url: System.fetch_env!("JINA_URL"))

      {:ok, docs} =
        ExAgent.embed(provider, chunks, task: :retrieval, args: [prompt_name: :document])

      {:ok, query} =
        ExAgent.embed(provider, "who supervises processes?",
          task: :retrieval,
          args: [prompt_name: :query]
        )

  It implements `c:ExAgent.Provider.embed/3` only; `chat/3` returns
  `{:error, %ExAgent.Error{type: :unsupported}}`.

  ## Behind Modal's proxy auth

      ExAgent.Providers.JinaV5.new(
        base_url: System.fetch_env!("MODAL_JINA_URL"),
        api_key: System.fetch_env!("MODAL_API_KEY"),
        headers: [
          {"Modal-Key", System.fetch_env!("MODAL_KEY")},
          {"Modal-Secret", System.fetch_env!("MODAL_SECRET")}
        ]
      )

  ## Tasks are v5's, not this library's

  v5 takes four task names, and the query/document distinction moved *out* of the
  task and into a separate `prompt_name`:

  | Task | `prompt_name` |
  |------|---------------|
  | `:retrieval` | **required** — `:query` for the search side, `:document` for the indexed side |
  | `:text_matching` | rejected |
  | `:clustering` | rejected |
  | `:classification` | rejected |

  This is why the module is named for the version. v3 and v4 spelled the same
  thing as a single `"retrieval.query"` / `"retrieval.passage"` task, so one
  provider covering both would have to lie about one of them.

  > #### Encode the two retrieval sides differently {: .warning}
  >
  > Asymmetric retrieval needs it. Embedding your documents *and* your queries
  > with the same `prompt_name` degrades recall invisibly, and the fix is a full
  > re-embed — so `:retrieval` without a `prompt_name` is an error here rather
  > than a default.

  ## Dimensions

  1024 by default, Matryoshka-truncatable to 32, 64, 128, 256, 512, or 768. The
  server truncates and re-normalizes; pass `args: [normalize: false]` to get the
  raw truncated vector instead, which is **not** unit length.

  ## The server contract

  Verified against a running deployment. `POST {base_url}/embed`:

      {"texts": ["..."], "task": "retrieval", "prompt_name": "document",
       "dimensions": 256, "normalize": true}

  answering

      {"model": "...", "task": "...", "prompt_name": "...",
       "dimensions": 256, "input_count": 1, "embeddings": [[...]]}

  Unknown body fields are rejected by the server, which is why `:args` is a
  closed allowlist rather than a passthrough. Batches are capped at 512 inputs.

  This is *not* the shape of Jina's hosted `api.jina.ai` service, which speaks an
  OpenAI-style `/v1/embeddings` with `model`/`input` and a `data[].embedding`
  response. Pointing this provider at the hosted API will not work.
  """

  @behaviour ExAgent.Provider

  # The key also lives in `:req`'s headers, and `:headers` is where gateway
  # credentials go; none of it belongs in a crash dump.
  @derive {Inspect, except: [:api_key, :headers, :req]}

  alias ExAgent.Error

  @default_model "jina-embeddings-v5-text-small"

  @type t :: %__MODULE__{
          base_url: String.t(),
          api_key: String.t() | nil,
          model: String.t(),
          headers: [{String.t(), String.t()}],
          req: Req.Request.t() | nil
        }

  @enforce_keys [:base_url]
  defstruct [:base_url, :api_key, :req, model: @default_model, headers: []]

  @schema [
    base_url: [
      type: :string,
      required: true,
      doc: "Root URL of the v5 server; requests go to `\#{base_url}/embed`"
    ],
    api_key: [
      type: {:or, [:string, nil]},
      default: nil,
      doc: "Sugar for an `Authorization: Bearer` header"
    ],
    model: [
      type: :string,
      default: @default_model,
      doc: "Recorded on the result when the server does not report its own"
    ],
    headers: [
      type: {:list, :any},
      default: [],
      doc: "Extra request headers, e.g. Modal's `Modal-Key` / `Modal-Secret`"
    ]
  ]

  @doc """
  Creates a provider with validated options and an initialized Req client.

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
  Returns the task atoms this provider accepts.

  ## Examples

      iex> ExAgent.Providers.JinaV5.embedding_tasks()
      [:retrieval, :text_matching, :clustering, :classification]
  """
  @spec embedding_tasks() :: [ExAgent.Embeddings.task()]
  defdelegate embedding_tasks, to: ExAgent.Services.JinaV5EmbedService, as: :tasks

  @impl true
  def chat(_provider, _messages, _opts \\ []) do
    {:error,
     Error.new(
       :unsupported,
       "JinaV5 is an embeddings model; use ExAgent.embed/3, or a chat provider " <>
         "such as ExAgent.Providers.OpenAI for conversations",
       __MODULE__
     )}
  end

  @impl true
  def embed(provider, inputs, opts \\ []) do
    ExAgent.Services.JinaV5EmbedService.embed(provider, inputs, opts)
  end

  @impl true
  def embedding_tasks(_provider), do: embedding_tasks()

  @spec build_req(t()) :: Req.Request.t()
  defp build_req(%__MODULE__{} = provider) do
    Req.new(base_url: provider.base_url, headers: auth(provider) ++ provider.headers)
  end

  @spec auth(t()) :: [{String.t(), String.t()}]
  defp auth(%__MODULE__{api_key: nil}), do: []
  defp auth(%__MODULE__{api_key: key}), do: [{"authorization", "Bearer #{key}"}]
end
