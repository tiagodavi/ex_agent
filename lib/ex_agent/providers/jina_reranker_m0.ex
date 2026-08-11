defmodule ExAgent.Providers.JinaRerankerM0 do
  @moduledoc """
  `jina-reranker-m0` - a **reranking-only** provider for a self-hosted server.

      provider = ExAgent.Providers.JinaRerankerM0.new(base_url: System.fetch_env!("RERANKER_URL"))

      {:ok, result} = ExAgent.rerank(provider, "who restarts processes?", chunks, top_n: 5)

      best = ExAgent.Reranking.take(result, chunks)

  It implements `c:ExAgent.Provider.rerank/4` only. `chat/3` returns
  `{:error, %ExAgent.Error{type: :unsupported}}`, and there is no `embed/3` - a
  reranker scores query/document *pairs* and has no single-text vector to give.

  ## Where it belongs in retrieval

  Reranking is the second stage, not a replacement for the first:

      {:ok, hits} = ExAgent.embed(embedder, [query], task: :retrieval, args: [prompt_name: :query])
      shortlist = vector_search(hits, limit: 100)

      {:ok, ranked} = ExAgent.rerank(reranker, query, shortlist, top_n: 10)

  Embeddings compare vectors computed independently, which is what makes them
  fast enough for a corpus. A cross-encoder reads the query and one document
  together - more accurate, and far too slow to run over everything. Batches are
  capped at 512 documents, which is the shape of the intended use.

  ## Behind Modal's proxy auth

      ExAgent.Providers.JinaRerankerM0.new(
        base_url: System.fetch_env!("MODAL_RERANKER_URL"),
        api_key: System.fetch_env!("MODAL_API_KEY"),
        headers: [
          {"Modal-Key", System.fetch_env!("MODAL_KEY")},
          {"Modal-Secret", System.fetch_env!("MODAL_SECRET")}
        ]
      )

  ## Scores

  Higher is more relevant, and that is the only guarantee - see
  `ExAgent.Reranking` on why an absolute cutoff does not port between models.

  ## The server contract

  Verified against a running deployment. `POST {base_url}/v1/rerank`:

      {"query": "...", "documents": ["..."], "top_n": 5, "return_documents": false}

  answering

      {"model": "...",
       "results": [{"index": 0, "relevance_score": 0.28, "document": {"text": "..."}}]}

  Unknown body fields are rejected by the server, so unrecognized options are
  reported rather than forwarded. `return_documents` defaults to `true` there and
  `false` here: `:index` already identifies each document, and sending a corpus
  back to have it re-quoted is waste.

  This is *not* the shape of Jina's hosted `api.jina.ai/v1/rerank` service, whose
  `documents` accept `{"text": ...}` / `{"image": ...}` objects for multimodal
  ranking. Pointing this provider there will not work.
  """

  @behaviour ExAgent.Provider

  # The key also lives in `:req`'s headers, and `:headers` is where gateway
  # credentials go; none of it belongs in a crash dump.
  @derive {Inspect, except: [:api_key, :headers, :req]}

  alias ExAgent.Error

  @default_model "jina-reranker-m0"

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
      doc: "Root URL of the reranker; requests go to `\#{base_url}/v1/rerank`"
    ],
    api_key: [
      type: {:or, [:string, nil]},
      default: nil,
      doc: "Sugar for an `Authorization: Bearer` header"
    ],
    model: [
      type: :string,
      default: @default_model,
      doc: "Sent in the request body and recorded on the result"
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

  @impl true
  def chat(_provider, _messages, _opts \\ []) do
    {:error,
     Error.new(
       :unsupported,
       "JinaRerankerM0 is a reranking model; use ExAgent.rerank/4, or a chat " <>
         "provider such as ExAgent.Providers.OpenAI for conversations",
       __MODULE__
     )}
  end

  @impl true
  def rerank(provider, query, documents, opts \\ []) do
    ExAgent.Services.JinaRerankerService.rerank(provider, query, documents, opts)
  end

  @spec build_req(t()) :: Req.Request.t()
  defp build_req(%__MODULE__{} = provider) do
    Req.new(base_url: provider.base_url, headers: auth(provider) ++ provider.headers)
  end

  @spec auth(t()) :: [{String.t(), String.t()}]
  defp auth(%__MODULE__{api_key: nil}), do: []
  defp auth(%__MODULE__{api_key: key}), do: [{"authorization", "Bearer #{key}"}]
end
