defmodule ExAgent do
  @moduledoc """
  Public API for the ExAgent multi-agent LLM library.

  One behaviour (`ExAgent.Provider`) abstracts OpenAI, Gemini, and any
  OpenAI-compatible endpoint; OTP primitives orchestrate them. The
  [README](readme.html) is the full guide with end-to-end recipes - this page is
  the API surface.

  ## Quick start

      provider = ExAgent.Providers.OpenAI.new(api_key: System.fetch_env!("OPENAI_API_KEY"))
      {:ok, agent} = ExAgent.start_agent(provider: provider)

      {:ok, response} = ExAgent.chat(agent, "What is Elixir?")
      response.content

  ## Roles

  Declare which provider serves which purpose in config, then name the purpose at
  the call site - see `ExAgent.Roles`.

      # config/runtime.exs
      config :ex_agent, :roles,
        chat: {ExAgent.Providers.Gemini, api_key: System.fetch_env!("GEMINI_API_KEY")}

      {:ok, agent} = ExAgent.start_agent(role: :chat)
      {:ok, response} = ExAgent.chat_with(:chat, "One-shot, no agent process")

  ## Conversations

  `chat/3` returns an `ExAgent.Response`. `chat_stream/3` yields `ExAgent.Chunk`
  structs and never raises. `collect/1` folds a chunk stream into the same
  `ExAgent.Response`, so streaming and blocking share one downstream path.

      {:ok, response} = agent |> ExAgent.chat_stream("Explain OTP") |> ExAgent.collect()

  Attach files with `files:` - `:path`, `:data`, `:url`, or `:file_ref`. Inline
  versus upload is decided per attachment against provider limits; see
  `ExAgent.Attachment`.

      ExAgent.chat(agent, "Describe this", files: [%{path: "photo.jpg"}])

  ## Embeddings

  Stateless, so `embed/3` takes a provider struct rather than an agent pid.
  Persist `model`, `dimensions`, and `task` alongside every vector - see
  `ExAgent.Embeddings`.

      {:ok, docs} = ExAgent.embed(gemini, chunks, task: :retrieval_document)

  Task vocabularies belong to the provider; `embedding_tasks/1` lists them.
  `ExAgent.Providers.JinaV5` is embeddings-only:

      {:ok, docs} =
        ExAgent.embed(jina, chunks, task: :retrieval, args: [prompt_name: :document])

  ## Reranking

  A reranker orders a shortlist the embeddings stage fetched, reading the query
  and each document together - see `ExAgent.Reranking`.

      {:ok, ranked} = ExAgent.rerank(reranker, question, shortlist, top_n: 10)

  ## Failures

  Every failed operation returns `{:error, %ExAgent.Error{}}` in one vocabulary,
  carrying a `:retryable?` flag, so retry logic is written once rather than per
  provider.

  ## Observability

  Every provider call emits `:telemetry` events - latency, token counts, and the
  normalized error type. See `ExAgent.Telemetry`.

  ## Agent architectures

  Eight patterns; the [README](readme.html#multi-agent-patterns) has a guide to
  choosing between them, with analogies and worked comparisons.

  *You* fix the order:

  - `ExAgent.Patterns.Chain` - a fixed sequence of steps, with gates between them

  The *input* picks the path:

  - `ExAgent.Patterns.Router` - match the input, dispatch in parallel, synthesize

  The *model* decides:

  - `ExAgent.Patterns.Subagents` - specialists invoked as tools, contexts isolated

  Control moves:

  - `ExAgent.Patterns.Handoff` - hand the conversation to another agent
  - `ExAgent.Patterns.Skills` - swap persona and tools inside one conversation

  Quality and scale:

  - `ExAgent.Patterns.Reflection` - draft, critique, revise until accepted
  - `ExAgent.Patterns.MapReduce` - split an oversized input, process, combine
  - `ExAgent.Patterns.Consensus` - sample several answers, go with the agreement
  """

  alias ExAgent.{Agent, Chunk, Context, FileRef, Message, Provider, Roles}
  alias ExAgent.Services.Structured
  alias ExAgent.Patterns.{Handoff, Router}

  # --- Provider Roles ---

  @doc """
  Returns the provider struct configured for `role`, raising if there is none.

  The struct is usable anywhere a hand-built one is - `start_agent/1`,
  `ExAgent.Provider.chat/3`, subagent specs, and every multi-agent pattern:

      {:ok, agent} = ExAgent.start_agent(provider: ExAgent.provider!(:chat))

  With `overrides`, returns a *fresh* struct built from the role's configured
  options with `overrides` merged over them - for per-tenant keys or per-request
  model selection:

      ExAgent.provider!(:chat, api_key: tenant.openai_key, model: "gpt-4o-mini")

  Overrides rebuild the `Req` client and are not cached, so this is fine per
  request but not inside a tight loop. See `ExAgent.Roles` for the config shape.
  """
  @spec provider!(atom(), keyword()) :: struct()
  def provider!(role, overrides \\ [])
  def provider!(role, []) when is_atom(role), do: Roles.fetch!(role)

  def provider!(role, overrides) when is_atom(role) and is_list(overrides),
    do: Roles.build_with!(role, overrides)

  @doc """
  Returns `{:ok, provider}` for `role`, or `:error` if it is not configured.

  The non-raising counterpart of `provider!/1`, for code that treats a missing
  role as a normal branch rather than a deployment bug.
  """
  @spec provider(atom()) :: {:ok, struct()} | :error
  defdelegate provider(role), to: Roles, as: :fetch

  @doc """
  Returns the configured role names, in declaration order.
  """
  @spec roles() :: [atom()]
  defdelegate roles, to: Roles, as: :list

  @doc """
  Sends a single message to the provider configured for `role`.

  Stateless: no agent process, no conversation history, no tool loop. If the
  role's provider carries `tools:`, a tool request comes back as the raw
  `{:tool_calls, calls}` from `c:ExAgent.Provider.chat/3` - **tools are not
  executed**. Use `start_agent(role: role, tools: [...])` when you want the loop.

  Accepts the same `:files` option as `chat/3`.

      {:ok, response} = ExAgent.chat_with(:vision, "What's the invoice total?",
        files: [%{url: "https://cdn.example.com/invoice.png"}])
  """
  @spec chat_with(atom(), String.t(), keyword()) ::
          {:ok, ExAgent.Response.t()}
          | {:tool_calls, [map()]}
          | {:error, ExAgent.Error.t()}
  def chat_with(role, input, opts \\ []) when is_atom(role) and is_binary(input) do
    with {:ok, message} <- user_message(input, opts) do
      Provider.chat(provider!(role), [message], opts)
    end
  end

  @doc """
  Streams a single message to the provider configured for `role`.

  Stateless, like `chat_with/3`, and with the same no-tool-loop caveat: any
  `:tool_call_delta` chunks are yielded but never executed.

  See `collect/1` to fold the result into a single `ExAgent.Response`.
  """
  @spec stream_with(atom(), String.t(), keyword()) :: Enumerable.t()
  def stream_with(role, input, opts \\ []) when is_atom(role) and is_binary(input) do
    case user_message(input, opts) do
      {:ok, message} -> Provider.stream(provider!(role), [message], opts)
      {:error, error} -> [Chunk.error(error)]
    end
  end

  @doc """
  Generates embeddings using the provider configured for `role`.

  Accepts the same options as `embed/3`. A role whose provider has no embeddings
  endpoint returns `{:error, %ExAgent.Error{type: :unsupported}}`.

      {:ok, docs} = ExAgent.embed_with(:embed, chunks, task: :retrieval_document)
  """
  @spec embed_with(atom(), [ExAgent.Embeddings.input()] | String.t(), keyword()) ::
          {:ok, ExAgent.Embeddings.t()} | {:error, ExAgent.Error.t()}
  def embed_with(role, inputs, opts \\ []) when is_atom(role),
    do: embed(provider!(role), inputs, opts)

  @spec user_message(String.t(), keyword()) ::
          {:ok, Message.t()} | {:error, ExAgent.Error.t()}
  defp user_message(input, opts) do
    case Message.new(role: :user, content: input, attachments: Keyword.get(opts, :files, [])) do
      {:ok, message} -> {:ok, message}
      {:error, reason} -> {:error, ExAgent.Error.new(:invalid_request, reason)}
    end
  end

  # --- Agent Lifecycle ---

  @doc """
  Starts a new agent under the dynamic supervisor.

  ## Options

  - `:provider` - struct implementing `ExAgent.Provider`
  - `:role` - role name resolved through `ExAgent.Roles`; mutually exclusive
    with `:provider`, and one of the two is required
  - `:id` - unique agent identifier
  - `:tools` - list of `ExAgent.Tool` structs
  - `:skills` - list of `ExAgent.Skill` structs
  - `:built_in_tools` - provider-specific built-in tools (e.g., `[:google_search]`, `[:web_search]`, `[:thinking]`)
  - `:name` - GenServer name for registration

  ## Examples

      provider = ExAgent.Providers.OpenAI.new(api_key: "sk-...")
      {:ok, pid} = ExAgent.start_agent(provider: provider)

      # or, with a role configured under `config :ex_agent, :roles`
      {:ok, pid} = ExAgent.start_agent(role: :vision)
  """
  @spec start_agent(Agent.agent_opts()) :: {:ok, pid()} | {:error, term()}
  defdelegate start_agent(opts), to: ExAgent.AgentDynamicSupervisor

  @doc """
  Stops an agent process.
  """
  @spec stop_agent(pid()) :: :ok | {:error, :not_found}
  defdelegate stop_agent(pid), to: ExAgent.AgentDynamicSupervisor

  # --- Chat ---

  @doc """
  Sends a user message to an agent and returns the response.

  ## Options

  - `:files` - list of file attachments, each a map with `:mime_type` and
    either `:data` (binary) or `:path` (file path). Files become part of
    the conversation context and are sent to the LLM alongside the text.
  - `:built_in_tools` - override agent-level built-in tools for this message.
    Examples: `[:google_search]`, `[:web_search]`, `[:thinking]`.

  ## Examples

      {:ok, response} = ExAgent.chat(agent, "What is Elixir?")
      response.content
      #=> "Elixir is a functional programming language..."

      # With file attachments
      {:ok, response} = ExAgent.chat(agent, "Describe this image",
        files: [%{path: "photo.jpg", mime_type: "image/jpeg"}])
  """
  @spec chat(GenServer.server(), String.t(), keyword()) ::
          {:ok, ExAgent.Response.t()}
          | {:handoff, pid() | atom(), Context.t()}
          | {:error, term()}
  defdelegate chat(agent, input, opts \\ []), to: Agent

  @doc """
  Sends a user message asynchronously, returning a Task.

  Accepts the same options as `chat/3`.
  """
  @spec chat_async(GenServer.server(), String.t(), keyword()) :: Task.t()
  defdelegate chat_async(agent, input, opts \\ []), to: Agent

  @doc """
  Streams an agent's response as a lazy `Stream` of `ExAgent.Chunk` structs.

  Only the final assistant turn is streamed; tool-call turns (if any) are
  resolved first. The conversation context is committed when the stream is fully
  consumed, so the returned stream must be consumed.

  Never raises - failures arrive as a terminal `:done` chunk carrying an
  `ExAgent.Error`.

  ## Examples

      agent
      |> ExAgent.chat_stream("Explain OTP supervision step by step")
      |> Enum.each(fn
        %ExAgent.Chunk{type: :text_delta, text: text} -> IO.write(text)
        _chunk -> :ok
      end)

  See `collect/1` to fold the stream into a single `ExAgent.Response`.
  """
  @spec chat_stream(GenServer.server(), String.t(), keyword()) :: Enumerable.t()
  defdelegate chat_stream(agent, input, opts \\ []), to: Agent

  # --- Context ---

  @doc """
  Returns the agent's current conversation context.
  """
  @spec get_context(GenServer.server()) :: Context.t()
  defdelegate get_context(agent), to: Agent

  @doc """
  Reset conversation context.
  """
  @spec reset(GenServer.server()) :: :ok
  defdelegate reset(agent), to: Agent

  # --- File Uploads ---

  @doc """
  Uploads a file from disk to the provider and returns a reference.

  The returned `FileRef` can be passed in chat messages via
  `files: [%{file_ref: ref}]` to avoid sending base64-encoded data inline.

  ## Options

  - `:filename` - override filename (defaults to basename of `file_path`)
  - `:purpose` - OpenAI-specific file purpose (default: `"user_data"`)

  ## Examples

      provider = ExAgent.Providers.OpenAI.new(api_key: "sk-...")
      {:ok, ref} = ExAgent.upload_file(provider, "report.pdf", "application/pdf")
      {:ok, response} = ExAgent.chat(agent, "Summarize", files: [%{file_ref: ref}])
  """
  @spec upload_file(struct(), String.t(), String.t(), keyword()) ::
          {:ok, FileRef.t()} | {:error, ExAgent.Error.t()}
  def upload_file(provider, file_path, mime_type, opts \\ []) do
    case File.read(file_path) do
      {:ok, data} ->
        opts = Keyword.put_new(opts, :filename, Path.basename(file_path))
        Provider.upload(provider, data, mime_type, opts)

      {:error, reason} ->
        {:error,
         ExAgent.Error.new(
           :invalid_request,
           "failed to read file #{file_path}: #{inspect(reason)}",
           provider.__struct__
         )}
    end
  end

  @doc """
  Uploads raw binary data to the provider and returns a reference.

  Use this when you already have file contents in memory.

  ## Options

  - `:filename` - filename for the upload (default: `"upload"`)
  - `:purpose` - OpenAI-specific file purpose (default: `"user_data"`)

  ## Examples

      image_bytes = File.read!("screenshot.png")
      {:ok, ref} = ExAgent.upload_data(provider, image_bytes, "image/png", filename: "screenshot.png")
  """
  @spec upload_data(struct(), binary(), String.t(), keyword()) ::
          {:ok, FileRef.t()} | {:error, ExAgent.Error.t()}
  def upload_data(provider, data, mime_type, opts \\ []) do
    Provider.upload(provider, data, mime_type, opts)
  end

  @doc """
  Folds a chunk stream into the same `ExAgent.Response` that `chat/3` returns.

  Lets streaming and non-streaming share one downstream code path: stream for
  the typing effect, then collect for the finished turn.

      {:ok, response} =
        agent
        |> ExAgent.chat_stream("Explain OTP")
        |> ExAgent.collect()

  Tool-call fragments are reassembled by index, and reasoning traces land in
  `:thinking` rather than `:content`. A stream that failed mid-flight returns
  its `{:error, %ExAgent.Error{}}`.

  Note that consuming a stream this way still commits the turn to the agent's
  context, exactly as iterating it would.

  ## Options

  - `:schema` - the same `Ecto` schema passed to `chat_stream/3`, which casts the
    finished text into `:structured`. It has to be given twice because a stream is
    a plain enumerable and carries no memory of how it was built. Passing it to
    `chat_stream/3` alone constrains the model but leaves `:structured` nil;
    passing it here alone casts an answer that was never constrained.

        {:ok, response} =
          agent
          |> ExAgent.chat_stream("Extract the invoice", schema: Invoice)
          |> ExAgent.collect(schema: Invoice)

        response.structured   #=> %Invoice{}

  There are no partial objects mid-stream: the JSON is decoded once, when the
  stream finishes.
  """
  @spec collect(Enumerable.t(), keyword()) ::
          {:ok, ExAgent.Response.t()} | {:error, ExAgent.Error.t()}
  def collect(chunks, opts \\ []) do
    acc =
      Enum.reduce(
        chunks,
        %{texts: [], thinking: [], calls: [], usage: %{}, finish: nil, error: nil},
        fn
          %Chunk{type: :text_delta, text: text}, acc when is_binary(text) ->
            %{acc | texts: [text | acc.texts]}

          %Chunk{type: :thinking_delta, text: text}, acc when is_binary(text) ->
            %{acc | thinking: [text | acc.thinking]}

          %Chunk{type: :tool_call_delta} = chunk, acc ->
            %{acc | calls: [chunk | acc.calls]}

          %Chunk{type: :usage, usage: usage}, acc when is_map(usage) ->
            %{acc | usage: Map.merge(acc.usage, usage)}

          %Chunk{type: :done, finish_reason: reason, error: error}, acc ->
            %{acc | finish: reason || acc.finish, error: error || acc.error}

          _chunk, acc ->
            acc
        end
      )

    case acc.error do
      nil -> Structured.decode({:ok, build_response(acc)}, opts[:schema], nil)
      error -> {:error, error}
    end
  end

  @spec build_response(map()) :: ExAgent.Response.t()
  defp build_response(acc) do
    {:ok, message} =
      Message.new(
        role: :assistant,
        content: acc.texts |> Enum.reverse() |> IO.iodata_to_binary(),
        tool_calls: acc.calls |> Enum.reverse() |> Chunk.finalize_tool_calls()
      )

    ExAgent.Response.new(message,
      thinking: collapse(acc.thinking),
      usage: acc.usage,
      finish_reason: acc.finish
    )
  end

  @spec collapse([String.t()]) :: String.t() | nil
  defp collapse([]), do: nil
  defp collapse(parts), do: parts |> Enum.reverse() |> IO.iodata_to_binary()

  # --- Embeddings ---

  @doc """
  Generates embedding vectors for one or more inputs.

  Takes a provider struct directly rather than an agent pid - embedding is
  stateless and has no conversation to carry.

  ## Options

  - `:model` - embedding model (each provider has its own default; this is never
    the provider's *chat* model)
  - `:dimensions` - output dimensionality, where the provider supports truncation
  - `:task` - what the embedding is for, as an atom from **that provider's** own
    vocabulary - see `embedding_tasks/1`. There is no shared set: Gemini's
    `taskType` enum, Jina v5's four task names, and OpenAI's absent task field
    have no common middle. An unknown task is an error naming the accepted ones,
    never a silently dropped field
  - `:args` - extra request-body parameters this library does not model, as a
    keyword list or map (`args: [prompt_name: :document]`). Each provider
    validates them against its own endpoint and rejects unknown keys

  ## Examples

      provider = ExAgent.Providers.Gemini.new(api_key: "AIza...")

      {:ok, docs} =
        ExAgent.embed(provider, ["Elixir is a functional language", "OTP supervises processes"],
          task: :retrieval_document
        )

      {:ok, query} = ExAgent.embed(provider, "what supervises processes?", task: :retrieval_query)

      [best | _] =
        docs.vectors
        |> Enum.map(&ExAgent.Embeddings.cosine_similarity(&1, hd(query.vectors)))
        |> Enum.with_index()
        |> Enum.sort_by(&elem(&1, 0), :desc)

  Persist `docs.model`, `docs.dimensions`, and `docs.task` alongside the vectors -
  see `ExAgent.Embeddings` for why.
  """
  @spec embed(struct(), [ExAgent.Embeddings.input()] | String.t(), keyword()) ::
          {:ok, ExAgent.Embeddings.t()} | {:error, ExAgent.Error.t()}
  def embed(provider, inputs, opts \\ [])

  def embed(provider, input, opts) when is_binary(input) or is_map(input),
    do: Provider.embed(provider, [input], opts)

  def embed(provider, inputs, opts) when is_list(inputs),
    do: Provider.embed(provider, inputs, opts)

  @doc """
  Reorders `documents` by relevance to `query`.

  Takes a provider struct directly, like `embed/3` - reranking is stateless.
  A provider with no reranking endpoint returns
  `{:error, %ExAgent.Error{type: :unsupported}}`.

  This is retrieval's **second** stage. Embeddings compare independently computed
  vectors, which is what makes searching a corpus feasible; a reranker reads the
  query and one document together, which is more accurate and far slower. Fetch a
  shortlist with the first, order it with the second.

  ## Options

  Provider-specific; see `ExAgent.Providers.JinaRerankerM0`. Commonly `:top_n`.

  ## Examples

      {:ok, result} = ExAgent.rerank(reranker, "who restarts processes?", chunks, top_n: 5)

      # `:index` points back into the list you passed, so it works for any record
      Enum.map(result.results, fn %{index: i, score: score} -> {Enum.at(rows, i), score} end)

  See `ExAgent.Reranking` for `take/2`, `above/2`, and why scores do not port
  between models.
  """
  @spec rerank(struct(), String.t(), [String.t()], keyword()) ::
          {:ok, ExAgent.Reranking.t()} | {:error, ExAgent.Error.t()}
  defdelegate rerank(provider, query, documents, opts \\ []), to: Provider

  @doc """
  Reranks using the provider configured for `role`.

  Accepts the same options as `rerank/4`.

      {:ok, ranked} = ExAgent.rerank_with(:reranker, question, shortlist, top_n: 10)
  """
  @spec rerank_with(atom(), String.t(), [String.t()], keyword()) ::
          {:ok, ExAgent.Reranking.t()} | {:error, ExAgent.Error.t()}
  def rerank_with(role, query, documents, opts \\ []) when is_atom(role),
    do: rerank(provider!(role), query, documents, opts)

  @doc """
  Returns the embedding task atoms `provider` accepts, or `[]` if it has none.

  Vocabularies are per provider, so this is how you discover one rather than
  guessing at a shared list:

      ExAgent.embedding_tasks(ExAgent.Providers.JinaV5.new())
      #=> [:retrieval, :text_matching, :clustering, :classification]

      ExAgent.embedding_tasks(ExAgent.Providers.OpenAI.new(api_key: "sk-..."))
      #=> []
  """
  @spec embedding_tasks(struct()) :: [ExAgent.Embeddings.task()]
  defdelegate embedding_tasks(provider), to: Provider

  # --- Patterns ---

  @doc """
  Routes input through matching agents and synthesizes results.

  See `ExAgent.Patterns.Router.run/2` for options.
  """
  @spec route(String.t(), Router.router_opts()) :: {:ok, String.t()} | {:error, term()}
  defdelegate route(input, opts), to: Router, as: :run

  @doc """
  Transfers conversation context to a target agent.
  """
  @spec handoff(pid() | atom(), Context.t()) :: :ok
  defdelegate handoff(target, context), to: Handoff, as: :run
end
