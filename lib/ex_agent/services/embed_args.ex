defmodule ExAgent.Services.EmbedArgs do
  @moduledoc false

  # Validation plumbing for the `:task` and `:args` embedding options.
  #
  # The *vocabularies* live in each provider's service - Gemini's `taskType` enum
  # and Jina v5's four task names have nothing in common, and pretending
  # otherwise is what the shared task map used to get wrong. What is genuinely
  # identical is the shape of the check and the wording of the failure, so only
  # that lives here.

  alias ExAgent.{Embeddings, Error}

  @doc false
  @spec validate_task(term(), [atom()], module()) :: :ok | {:error, Error.t()}
  def validate_task(nil, _allowed, _provider), do: :ok

  def validate_task(task, allowed, provider) when is_atom(task) do
    if task in allowed do
      :ok
    else
      {:error,
       Error.new(
         :invalid_request,
         "unknown task #{inspect(task)} for #{name(provider)}; expected one of " <>
           "#{inspect(allowed)}",
         provider
       )}
    end
  end

  # A string would be forwarded verbatim and accepted with a 200 by an endpoint
  # that does not recognize it, leaving quietly wrong vectors in an index.
  def validate_task(task, allowed, provider) do
    {:error,
     Error.new(
       :invalid_request,
       "task must be an atom from #{name(provider)}'s vocabulary #{inspect(allowed)}, " <>
         "got #{inspect(task)}",
       provider
     )}
  end

  @doc false
  # `allowed` maps each accepted key to a list of accepted values, or to `:any`
  # where the endpoint takes free-form input.
  @spec validate_args(term(), keyword(), module()) :: {:ok, keyword()} | {:error, Error.t()}
  def validate_args(args, allowed, provider) do
    with {:ok, pairs} <- normalize(args, provider) do
      Enum.reduce_while(pairs, {:ok, []}, fn {key, value}, {:ok, acc} ->
        case validate_pair(key, value, allowed, provider) do
          {:ok, pair} -> {:cont, {:ok, [pair | acc]}}
          {:error, _reason} = failure -> {:halt, failure}
        end
      end)
      |> case do
        {:ok, validated} -> {:ok, Enum.reverse(validated)}
        {:error, _reason} = failure -> failure
      end
    end
  end

  @spec normalize(term(), module()) :: {:ok, keyword()} | {:error, Error.t()}
  defp normalize(args, provider) do
    case Embeddings.normalize_args(args) do
      {:ok, pairs} ->
        {:ok, pairs}

      :error ->
        {:error,
         Error.new(
           :invalid_request,
           ":args must be a keyword list or a map, got #{inspect(args)}",
           provider
         )}
    end
  end

  @spec validate_pair(atom(), term(), keyword(), module()) ::
          {:ok, {atom(), term()}} | {:error, Error.t()}
  defp validate_pair(key, value, allowed, provider) do
    case Keyword.fetch(allowed, key) do
      :error -> {:error, unknown_key(key, allowed, provider)}
      {:ok, :any} -> {:ok, {key, value}}
      {:ok, values} -> check_value(key, value, values, provider)
    end
  end

  # Atoms are accepted wherever the endpoint wants one of a fixed set of strings:
  # `prompt_name: :document` reads better in Elixir than `"document"`, and there
  # is no ambiguity to preserve.
  @spec check_value(atom(), term(), [term()], module()) ::
          {:ok, {atom(), term()}} | {:error, Error.t()}
  defp check_value(key, value, values, provider) do
    normalized = if is_atom(value) and not is_boolean(value), do: to_string(value), else: value

    if normalized in values do
      {:ok, {key, normalized}}
    else
      {:error,
       Error.new(
         :invalid_request,
         "invalid value #{inspect(value)} for :#{key} on #{name(provider)}; " <>
           "expected one of #{inspect(values)}",
         provider
       )}
    end
  end

  @spec unknown_key(atom(), keyword(), module()) :: Error.t()
  defp unknown_key(key, [], provider) do
    Error.new(
      :invalid_request,
      "#{name(provider)} accepts no extra :args, got :#{key}",
      provider
    )
  end

  defp unknown_key(key, allowed, provider) do
    Error.new(
      :invalid_request,
      "#{name(provider)} does not accept :#{key} in :args; it accepts " <>
        "#{inspect(Keyword.keys(allowed))}",
      provider
    )
  end

  @spec name(module()) :: String.t()
  defp name(provider), do: provider |> Module.split() |> List.last()
end
