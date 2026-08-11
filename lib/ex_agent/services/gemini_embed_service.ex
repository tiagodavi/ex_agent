defmodule ExAgent.Services.GeminiEmbedService do
  @moduledoc """
  HTTP service for the Gemini embeddings API.

  Gemini's embedding models express the task in two incompatible ways, so the
  model name selects a *family*:

  - `:task_type` (`gemini-embedding-001`, `text-embedding-004`) — the task is a
    `taskType` enum field on the request, and truncated output is **not**
    normalized, so it is scaled client-side.
  - `:prefix` (`gemini-embedding-2`) — there is no `taskType` field; the task is
    written into the text itself, and truncated output is self-normalizing.

  Requests always go through `batchEmbedContents` with one `Content` per input.
  A flat multi-input list returns a single *aggregated* vector on
  `gemini-embedding-2`, which would silently corrupt an entire index.
  """

  alias ExAgent.Providers.Gemini
  alias ExAgent.{Embeddings, Error}

  @default_model "gemini-embedding-001"
  @full_dimensions 3072

  @task_types %{
    retrieval_query: "RETRIEVAL_QUERY",
    retrieval_document: "RETRIEVAL_DOCUMENT",
    similarity: "SEMANTIC_SIMILARITY",
    classification: "CLASSIFICATION",
    clustering: "CLUSTERING",
    question_answering: "QUESTION_ANSWERING",
    fact_verification: "FACT_VERIFICATION",
    code_query: "CODE_RETRIEVAL_QUERY"
  }

  # `gemini-embedding-2` has no taskType field, so the task is encoded as a text
  # prefix instead.
  @query_prefixes %{
    retrieval_query: "search result",
    question_answering: "question answering",
    fact_verification: "fact checking",
    code_query: "code retrieval",
    classification: "classification",
    clustering: "clustering",
    similarity: "sentence similarity"
  }

  @doc """
  Generates embeddings for `inputs`.

  ## Options

  - `:model` - defaults to `#{inspect(@default_model)}`. Never the provider's chat model.
  - `:dimensions` - sets `outputDimensionality`
  - `:task` - one of `ExAgent.Embeddings.tasks/0`
  - `:embedding_family` - `:task_type` or `:prefix`, to use a model this library
    does not know yet
  """
  @spec embed(Gemini.t(), [Embeddings.input()], keyword()) ::
          {:ok, Embeddings.t()} | {:error, Error.t()}
  def embed(%Gemini{} = provider, inputs, opts \\ []) do
    model = opts[:model] || @default_model
    task = opts[:task]
    dimensions = opts[:dimensions]

    with {:ok, family} <- family(model, opts[:embedding_family]),
         :ok <- validate_task(task),
         {:ok, normalized} <- normalize_inputs(inputs) do
      body = %{
        "requests" => Enum.map(normalized, &build_request(family, &1, model, task, dimensions))
      }

      provider.req
      |> Req.post(
        url: "/models/#{model}:batchEmbedContents",
        json: body,
        receive_timeout: :timer.minutes(5)
      )
      |> Error.from_result(Gemini)
      |> case do
        {:ok, response} -> build_result(response, family, model, task, dimensions)
        {:error, _error} = failure -> failure
      end
    end
  end

  # Fail loudly on an unknown model. Guessing is worse than erroring: sending a
  # taskType to a model that ignores it — or prefixing text for a model that does
  # not expect it — returns HTTP 200 with plausible floats that land in a vector
  # store and quietly degrade retrieval forever. There is no later signal.
  @spec family(String.t(), atom() | nil) :: {:ok, :task_type | :prefix} | {:error, Error.t()}
  defp family(_model, override) when override in [:task_type, :prefix], do: {:ok, override}

  defp family("gemini-embedding-001" <> _rest, _override), do: {:ok, :task_type}
  defp family("text-embedding-004" <> _rest, _override), do: {:ok, :task_type}
  defp family("gemini-embedding-2" <> _rest, _override), do: {:ok, :prefix}

  defp family(model, _override) do
    {:error,
     Error.new(
       :unsupported,
       "unknown Gemini embedding model #{inspect(model)}; pass " <>
         ":embedding_family (:task_type or :prefix) to use a model this version " <>
         "does not know about",
       Gemini
     )}
  end

  @spec validate_task(Embeddings.task_input() | nil) :: :ok | {:error, Error.t()}
  defp validate_task(nil), do: :ok

  # Gemini's taskType is a closed enum, so a raw string is a typo far more often
  # than a new value — and `:embedding_family` already covers model drift.
  defp validate_task(task) when is_binary(task) do
    {:error,
     Error.new(
       :invalid_request,
       "Gemini takes a normalized task atom, not the string #{inspect(task)}; " <>
         "expected one of #{inspect(Embeddings.tasks())}. Verbatim task strings are " <>
         "an OpenAI-compatible-endpoint feature, where the vocabulary is model-specific",
       Gemini
     )}
  end

  defp validate_task(task) do
    if Embeddings.valid_task?(task) do
      :ok
    else
      {:error,
       Error.new(
         :invalid_request,
         "unknown task #{inspect(task)}; expected one of #{inspect(Embeddings.tasks())}",
         Gemini
       )}
    end
  end

  @spec normalize_inputs([Embeddings.input()]) ::
          {:ok, [%{content: String.t(), title: String.t() | nil}]} | {:error, Error.t()}
  defp normalize_inputs(inputs) do
    Enum.reduce_while(inputs, {:ok, []}, fn
      text, {:ok, acc} when is_binary(text) ->
        {:cont, {:ok, [%{content: text, title: nil} | acc]}}

      %{content: content} = input, {:ok, acc} when is_binary(content) ->
        {:cont, {:ok, [%{content: content, title: Map.get(input, :title)} | acc]}}

      other, {:ok, _acc} ->
        {:halt,
         {:error,
          Error.new(
            :invalid_request,
            "each input must be a string or a map with :content, got #{inspect(other)}",
            Gemini
          )}}
    end)
    |> case do
      {:ok, normalized} -> {:ok, Enum.reverse(normalized)}
      {:error, _} = err -> err
    end
  end

  @spec build_request(
          :task_type | :prefix,
          map(),
          String.t(),
          Embeddings.task() | nil,
          pos_integer() | nil
        ) ::
          map()
  defp build_request(:task_type, input, model, task, dimensions) do
    %{
      "model" => "models/#{model}",
      "content" => %{"parts" => [%{"text" => input.content}]}
    }
    |> maybe_put("taskType", task_type(task))
    |> maybe_put("outputDimensionality", dimensions)
  end

  defp build_request(:prefix, input, model, task, dimensions) do
    %{
      "model" => "models/#{model}",
      "content" => %{"parts" => [%{"text" => apply_prefix(input, task)}]}
    }
    |> maybe_put("outputDimensionality", dimensions)
  end

  @spec task_type(Embeddings.task() | nil) :: String.t() | nil
  defp task_type(nil), do: nil
  defp task_type(task), do: @task_types[task]

  @spec apply_prefix(map(), Embeddings.task() | nil) :: String.t()
  defp apply_prefix(%{content: content}, nil), do: content

  defp apply_prefix(%{content: content, title: title}, :retrieval_document),
    do: "title: #{title || "none"} | text: #{content}"

  defp apply_prefix(%{content: content}, task) do
    case @query_prefixes[task] do
      nil -> content
      prefix -> "task: #{prefix} | query: #{content}"
    end
  end

  @spec build_result(
          map(),
          :task_type | :prefix,
          String.t(),
          Embeddings.task() | nil,
          pos_integer() | nil
        ) :: {:ok, Embeddings.t()} | {:error, Error.t()}
  defp build_result(%{"embeddings" => embeddings}, family, model, task, dimensions)
       when is_list(embeddings) do
    vectors =
      embeddings
      |> Enum.map(&Map.get(&1, "values", []))
      |> post_process(family, dimensions)

    {:ok,
     %Embeddings{
       vectors: vectors,
       model: model,
       provider: Gemini,
       dimensions: dimensions || dimensions_of(vectors),
       task: task,
       # batchEmbedContents returns no token counts.
       usage: %{}
     }}
  end

  defp build_result(body, _family, _model, _task, _dimensions),
    do: {:error, Error.unexpected_response(body, Gemini)}

  # `gemini-embedding-001` does not renormalize a truncated vector, but cosine
  # similarity assumes unit length. `gemini-embedding-2` normalizes its own.
  @spec post_process([[float()]], :task_type | :prefix, pos_integer() | nil) :: [[float()]]
  defp post_process(vectors, :task_type, dimensions)
       when is_integer(dimensions) and dimensions != @full_dimensions,
       do: Enum.map(vectors, &Embeddings.l2_normalize/1)

  defp post_process(vectors, _family, _dimensions), do: vectors

  @spec dimensions_of([[float()]]) :: pos_integer() | nil
  defp dimensions_of([first | _]) when is_list(first) and first != [], do: length(first)
  defp dimensions_of(_vectors), do: nil

  @spec maybe_put(map(), String.t(), term()) :: map()
  defp maybe_put(body, _key, nil), do: body
  defp maybe_put(body, key, value), do: Map.put(body, key, value)
end
