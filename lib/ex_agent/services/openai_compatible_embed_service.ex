defmodule ExAgent.Services.OpenAICompatibleEmbedService do
  @moduledoc """
  HTTP service for embeddings on OpenAI-compatible endpoints (vLLM, Jina, ...).

  Unlike OpenAI proper, these endpoints commonly accept `task` and `dimensions`
  as body fields. The task *strings* are model-specific and have changed between
  model versions, so the default map is overridable per call via `:task_map`.
  """

  alias ExAgent.Providers.OpenAICompatible
  alias ExAgent.{Embeddings, Error}

  # Jina-style defaults; override with :task_map for other models.
  @default_task_map %{
    retrieval_query: "retrieval.query",
    retrieval_document: "retrieval.passage",
    similarity: "text-matching",
    classification: "classification",
    clustering: "separation",
    question_answering: "retrieval.query",
    fact_verification: "retrieval.query",
    code_query: "code.query"
  }

  @doc """
  Generates embeddings for `inputs`.

  ## Options

  - `:model` - defaults to the provider's `:model`
  - `:dimensions` - sent as a `dimensions` body field
  - `:task` - translated through `:task_map` and sent as a `task` body field
  - `:task_map` - override the task-string translation for this model
  """
  @spec embed(OpenAICompatible.t(), [Embeddings.input()], keyword()) ::
          {:ok, Embeddings.t()} | {:error, Error.t()}
  def embed(%OpenAICompatible{} = provider, inputs, opts \\ []) do
    model = opts[:model] || provider.model
    task = opts[:task]
    dimensions = opts[:dimensions]
    task_map = Map.merge(@default_task_map, opts[:task_map] || %{})

    with {:ok, texts} <- to_texts(inputs),
         {:ok, task_string} <- translate_task(task, task_map) do
      body =
        %{"model" => model, "input" => texts}
        |> maybe_put("dimensions", dimensions)
        |> maybe_put("task", task_string)

      provider.req
      |> Req.post(url: "/embeddings", json: body, receive_timeout: :timer.minutes(5))
      |> Error.from_result(OpenAICompatible)
      |> case do
        {:ok, response} -> build_result(response, model, task, dimensions)
        {:error, _error} = failure -> failure
      end
    end
  end

  @doc false
  @spec default_task_map() :: %{Embeddings.task() => String.t()}
  def default_task_map, do: @default_task_map

  @spec translate_task(Embeddings.task() | nil, map()) ::
          {:ok, String.t() | nil} | {:error, Error.t()}
  defp translate_task(nil, _task_map), do: {:ok, nil}

  defp translate_task(task, task_map) do
    case Map.fetch(task_map, task) do
      {:ok, string} ->
        {:ok, string}

      :error ->
        {:error,
         Error.new(
           :unsupported,
           "no task string for #{inspect(task)} on this endpoint; pass :task_map to " <>
             "supply one for your model",
           OpenAICompatible
         )}
    end
  end

  @spec to_texts([Embeddings.input()]) :: {:ok, [String.t()]} | {:error, Error.t()}
  defp to_texts(inputs) do
    Enum.reduce_while(inputs, {:ok, []}, fn
      text, {:ok, acc} when is_binary(text) ->
        {:cont, {:ok, [text | acc]}}

      %{content: content}, {:ok, acc} when is_binary(content) ->
        {:cont, {:ok, [content | acc]}}

      other, {:ok, _acc} ->
        {:halt,
         {:error,
          Error.new(
            :invalid_request,
            "each input must be a string or a map with :content, got #{inspect(other)}",
            OpenAICompatible
          )}}
    end)
    |> case do
      {:ok, texts} -> {:ok, Enum.reverse(texts)}
      {:error, _} = err -> err
    end
  end

  @spec build_result(map(), String.t(), Embeddings.task() | nil, pos_integer() | nil) ::
          {:ok, Embeddings.t()} | {:error, Error.t()}
  defp build_result(%{"data" => data} = body, model, task, dimensions) when is_list(data) do
    vectors =
      data
      |> Enum.sort_by(&Map.get(&1, "index", 0))
      |> Enum.map(&Map.get(&1, "embedding", []))

    {:ok,
     %Embeddings{
       vectors: vectors,
       model: model,
       provider: OpenAICompatible,
       dimensions: dimensions || dimensions_of(vectors),
       task: task,
       usage: usage_map(body["usage"])
     }}
  end

  defp build_result(body, _model, _task, _dimensions),
    do: {:error, Error.unexpected_response(body, OpenAICompatible)}

  @spec usage_map(term()) :: map()
  defp usage_map(%{} = usage) do
    %{
      input_tokens: usage["prompt_tokens"],
      total_tokens: usage["total_tokens"]
    }
  end

  defp usage_map(_usage), do: %{}

  @spec dimensions_of([[float()]]) :: pos_integer() | nil
  defp dimensions_of([first | _]) when is_list(first) and first != [], do: length(first)
  defp dimensions_of(_vectors), do: nil

  @spec maybe_put(map(), String.t(), term()) :: map()
  defp maybe_put(body, _key, nil), do: body
  defp maybe_put(body, key, value), do: Map.put(body, key, value)
end
