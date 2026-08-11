defmodule ExAgent.Services.OpenAIEmbedService do
  @moduledoc """
  HTTP service for the OpenAI embeddings API.

  OpenAI supports `:dimensions` but has no notion of a task type. A non-nil
  `:task` is rejected rather than dropped — a silently discarded task produces
  vectors that look fine and retrieve badly.
  """

  alias ExAgent.Providers.OpenAI
  alias ExAgent.Services.EmbedArgs
  alias ExAgent.{Embeddings, Error}

  @default_model "text-embedding-3-small"

  # OpenAI rejects a request with more than 2048 inputs. Chunking is silent
  # magic; a clear error naming the limit is not, and the caller knows how they
  # want to batch.
  @max_inputs 2048

  # OpenAI has no task field at all, so there is no vocabulary to declare.
  @tasks []

  @allowed_args [encoding_format: ["float", "base64"], user: :any]

  @doc """
  Returns the task atoms this service accepts — none.
  """
  @spec tasks() :: [Embeddings.task()]
  def tasks, do: @tasks

  @doc """
  Generates embeddings for `inputs`.

  ## Options

  - `:model` - defaults to `#{inspect(@default_model)}`
  - `:dimensions` - truncate output to this many dimensions
  - `:task` - unsupported; any non-nil value returns an error
  - `:args` - `encoding_format` (`"float"` or `"base64"`) and `user`; any other
    key is rejected
  """

  @spec embed(OpenAI.t(), [Embeddings.input()], keyword()) ::
          {:ok, Embeddings.t()} | {:error, Error.t()}
  def embed(%OpenAI{} = provider, inputs, opts \\ []) do
    model = opts[:model] || @default_model
    dimensions = opts[:dimensions]

    with :ok <- reject_task(opts[:task]),
         :ok <- check_batch_size(inputs),
         {:ok, args} <- EmbedArgs.validate_args(opts[:args], @allowed_args, OpenAI),
         {:ok, texts} <- to_texts(inputs) do
      body =
        %{"model" => model, "input" => texts}
        |> maybe_put("dimensions", dimensions)
        |> merge_args(args)

      provider.req
      |> Req.post(url: "/embeddings", json: body, receive_timeout: :timer.minutes(5))
      |> Error.from_result(OpenAI)
      |> case do
        {:ok, response} -> build_result(response, model, dimensions)
        {:error, _error} = failure -> failure
      end
    end
  end

  @spec check_batch_size([Embeddings.input()]) :: :ok | {:error, Error.t()}
  defp check_batch_size(inputs) do
    count = length(inputs)

    if count > @max_inputs do
      {:error,
       Error.new(
         :invalid_request,
         "OpenAI accepts at most #{@max_inputs} inputs per embeddings request, got #{count}; " <>
           "split them with Enum.chunk_every/2",
         OpenAI
       )}
    else
      :ok
    end
  end

  @spec merge_args(map(), keyword()) :: map()
  defp merge_args(body, args) do
    Enum.reduce(args, body, fn {key, value}, acc -> Map.put(acc, to_string(key), value) end)
  end

  @spec reject_task(Embeddings.task() | nil) :: :ok | {:error, Error.t()}
  defp reject_task(nil), do: :ok

  defp reject_task(task) do
    {:error,
     Error.new(
       :unsupported,
       "OpenAI embeddings have no task type (got #{inspect(task)}). Omit :task, or use " <>
         "Gemini or an OpenAI-compatible endpoint if you need task-conditioned vectors",
       OpenAI
     )}
  end

  # Text only: a map input carries a :title, which has nowhere to go here.
  @spec to_texts([Embeddings.input()]) :: {:ok, [String.t()]} | {:error, Error.t()}
  defp to_texts(inputs) do
    Enum.reduce_while(inputs, {:ok, []}, fn
      text, {:ok, acc} when is_binary(text) ->
        {:cont, {:ok, [text | acc]}}

      other, {:ok, _acc} ->
        {:halt,
         {:error,
          Error.new(
            :invalid_request,
            "OpenAI embeddings accept text only, got #{inspect(other)}",
            OpenAI
          )}}
    end)
    |> case do
      {:ok, texts} -> {:ok, Enum.reverse(texts)}
      {:error, _} = err -> err
    end
  end

  @spec build_result(map(), String.t(), pos_integer() | nil) ::
          {:ok, Embeddings.t()} | {:error, Error.t()}
  defp build_result(%{"data" => data} = body, model, dimensions) when is_list(data) do
    # The API does not guarantee response order; out-of-order vectors would
    # silently mis-associate every embedding with the wrong input.
    vectors =
      data
      |> Enum.sort_by(&Map.get(&1, "index", 0))
      |> Enum.map(&Map.get(&1, "embedding", []))

    {:ok,
     %Embeddings{
       vectors: vectors,
       model: model,
       provider: OpenAI,
       dimensions: dimensions || dimensions_of(vectors),
       task: nil,
       usage: usage_map(body["usage"])
     }}
  end

  defp build_result(body, _model, _dimensions),
    do: {:error, Error.unexpected_response(body, OpenAI)}

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
