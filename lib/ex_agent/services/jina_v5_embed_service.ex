defmodule ExAgent.Services.JinaV5EmbedService do
  @moduledoc """
  HTTP service for a self-hosted Jina embeddings v5 server.

  ## The contract

  `POST {base_url}/embed`, with the server's own field names:

      {"texts": ["..."], "task": "retrieval", "prompt_name": "document",
       "dimensions": 256, "normalize": true}

  answering

      {"model": "...", "task": "...", "prompt_name": "...",
       "dimensions": 256, "input_count": 1, "embeddings": [[...]]}

  Every rule below was checked against a running deployment, not inferred from a
  model card - the server rejects unknown fields outright, so guessing is not an
  option.
  """

  alias ExAgent.Providers.JinaV5
  alias ExAgent.Services.EmbedArgs
  alias ExAgent.{Embeddings, Error}

  @tasks ~w(retrieval text_matching clustering classification)a

  # v5's own spelling for each task name.
  @task_strings %{
    retrieval: "retrieval",
    text_matching: "text-matching",
    clustering: "clustering",
    classification: "classification"
  }

  @default_task :retrieval

  # Matryoshka checkpoints. Anything else is refused by the server.
  @dimensions [32, 64, 128, 256, 512, 768, 1024]

  # `texts` is capped server-side; erroring here saves a round trip and names the
  # limit instead of returning a validation blob.
  @max_inputs 512

  @prompt_names ["query", "document"]

  # The server forbids unknown fields, so this allowlist is exhaustive rather
  # than a convenience.
  @allowed_args [prompt_name: @prompt_names, normalize: [true, false]]

  @doc """
  Returns the task atoms this service accepts.
  """
  @spec tasks() :: [Embeddings.task()]
  def tasks, do: @tasks

  @doc """
  Generates embeddings for `inputs`.

  ## Options

  - `:dimensions` - one of `#{inspect(@dimensions)}`; the server truncates and
    re-normalizes
  - `:task` - one of `#{inspect(@tasks)}`; defaults to `#{inspect(@default_task)}`,
    matching the server
  - `:args` - `prompt_name: :query | :document` and `normalize: true | false`.
    Nothing else: the server rejects unknown fields

  ## Examples

      ExAgent.embed(provider, chunks, task: :retrieval, args: [prompt_name: :document])
  """
  @spec embed(JinaV5.t(), [Embeddings.input()], keyword()) ::
          {:ok, Embeddings.t()} | {:error, Error.t()}
  def embed(%JinaV5{} = provider, inputs, opts \\ []) do
    task = Keyword.get(opts, :task, @default_task)
    dimensions = opts[:dimensions]

    with :ok <- EmbedArgs.validate_task(task, @tasks, JinaV5),
         :ok <- validate_dimensions(dimensions),
         {:ok, texts} <- to_texts(inputs),
         :ok <- validate_batch(texts),
         {:ok, args} <- EmbedArgs.validate_args(opts[:args], @allowed_args, JinaV5),
         :ok <- validate_prompt_name(task, args) do
      body =
        %{"texts" => texts, "task" => @task_strings[task]}
        |> maybe_put("dimensions", dimensions)
        |> merge_args(args)

      provider.req
      |> Req.post(url: "/embed", json: body, receive_timeout: :timer.minutes(5))
      |> Error.from_result(JinaV5)
      |> case do
        {:ok, response} -> build_result(response, provider, task, dimensions)
        {:error, _error} = failure -> failure
      end
    end
  end

  # The server requires a `prompt_name` for retrieval and forbids one elsewhere.
  # Both are enforced here so the failure names the fix rather than arriving as a
  # 400 after the request.
  @spec validate_prompt_name(Embeddings.task(), keyword()) :: :ok | {:error, Error.t()}
  defp validate_prompt_name(:retrieval, args) do
    if Keyword.has_key?(args, :prompt_name) do
      :ok
    else
      {:error,
       Error.new(
         :invalid_request,
         "task :retrieval needs args: [prompt_name: :query] for the search side or " <>
           "[prompt_name: :document] for the indexed side. Encoding both sides the same " <>
           "way degrades recall with no error to notice",
         JinaV5
       )}
    end
  end

  defp validate_prompt_name(task, args) do
    if Keyword.has_key?(args, :prompt_name) do
      {:error,
       Error.new(
         :invalid_request,
         ":prompt_name applies only to task :retrieval, not #{inspect(task)}",
         JinaV5
       )}
    else
      :ok
    end
  end

  @spec validate_dimensions(term()) :: :ok | {:error, Error.t()}
  defp validate_dimensions(nil), do: :ok
  defp validate_dimensions(dimensions) when dimensions in @dimensions, do: :ok

  defp validate_dimensions(dimensions) do
    {:error,
     Error.new(
       :invalid_request,
       "Jina v5 is trained for #{inspect(@dimensions)} dimensions, got " <>
         "#{inspect(dimensions)}; an untrained width retrieves badly",
       JinaV5
     )}
  end

  @spec validate_batch([String.t()]) :: :ok | {:error, Error.t()}
  defp validate_batch(texts) when length(texts) <= @max_inputs, do: :ok

  defp validate_batch(texts) do
    {:error,
     Error.new(
       :invalid_request,
       "at most #{@max_inputs} inputs per request, got #{length(texts)}; " <>
         "split them with Enum.chunk_every/2",
       JinaV5
     )}
  end

  @spec merge_args(map(), keyword()) :: map()
  defp merge_args(body, args) do
    Enum.reduce(args, body, fn {key, value}, acc -> Map.put(acc, to_string(key), value) end)
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
            JinaV5
          )}}
    end)
    |> case do
      {:ok, texts} -> {:ok, Enum.reverse(texts)}
      {:error, _reason} = failure -> failure
    end
  end

  # Vectors come back in request order, and the server owns normalization -
  # re-normalizing here would silently undo an explicit `normalize: false`.
  @spec build_result(map(), JinaV5.t(), Embeddings.task(), pos_integer() | nil) ::
          {:ok, Embeddings.t()} | {:error, Error.t()}
  defp build_result(%{"embeddings" => vectors} = body, provider, task, dimensions)
       when is_list(vectors) do
    {:ok,
     %Embeddings{
       vectors: vectors,
       model: body["model"] || provider.model,
       provider: JinaV5,
       dimensions: body["dimensions"] || dimensions || dimensions_of(vectors),
       task: task,
       # The server reports no token counts.
       usage: %{}
     }}
  end

  defp build_result(body, _provider, _task, _dimensions),
    do: {:error, Error.unexpected_response(body, JinaV5)}

  @spec dimensions_of([[float()]]) :: pos_integer() | nil
  defp dimensions_of([first | _rest]) when is_list(first) and first != [], do: length(first)
  defp dimensions_of(_vectors), do: nil

  @spec maybe_put(map(), String.t(), term()) :: map()
  defp maybe_put(body, _key, nil), do: body
  defp maybe_put(body, key, value), do: Map.put(body, key, value)
end
