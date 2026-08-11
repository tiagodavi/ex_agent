defmodule ExAgent.Services.JinaRerankerService do
  @moduledoc """
  HTTP service for a self-hosted `jina-reranker-m0` server.

  ## The contract

  `POST {base_url}/v1/rerank`:

      {"query": "...", "documents": ["...", "..."],
       "top_n": 5, "return_documents": false}

  answering

      {"model": "...",
       "results": [{"index": 0, "relevance_score": 0.28,
                    "document": {"text": "..."}}]}

  Results arrive ordered by descending score. `document` is nested under a
  `"text"` key and only present when asked for; `return_documents` defaults to
  `true` server-side, and this service sends `false` unless you opt in, because
  `:index` already identifies the document and echoing a corpus back is waste.

  Verified against a running deployment: the server rejects unknown body fields,
  requires at least one document, and caps a batch at 512.
  """

  alias ExAgent.Providers.JinaRerankerM0
  alias ExAgent.{Error, Reranking}

  # Enforced server-side; erroring here names the limit instead of returning a
  # validation blob, and saves the round trip.
  @max_documents 512

  @known_opts [:model, :top_n, :return_documents]

  @doc """
  Reorders `documents` by relevance to `query`.

  ## Options

  - `:model` - overrides the provider's `:model` in the request body
  - `:top_n` - return only the best `n`; the server still scores every document
  - `:return_documents` - echo the document text back (default `false`)

  ## Examples

      ExAgent.rerank(provider, "who restarts processes?", chunks, top_n: 5)
  """
  @spec rerank(JinaRerankerM0.t(), String.t(), [String.t()], keyword()) ::
          {:ok, Reranking.t()} | {:error, Error.t()}
  def rerank(%JinaRerankerM0{} = provider, query, documents, opts \\ []) do
    with :ok <- validate_query(query),
         :ok <- validate_documents(documents),
         :ok <- validate_opts(opts),
         :ok <- validate_top_n(opts[:top_n]) do
      body = build_body(provider, query, documents, opts)

      provider.req
      |> Req.post(url: "/v1/rerank", json: body, receive_timeout: :timer.minutes(5))
      |> Error.from_result(JinaRerankerM0)
      |> case do
        {:ok, response} -> build_result(response, provider, length(documents))
        {:error, _error} = failure -> failure
      end
    end
  end

  @spec build_body(JinaRerankerM0.t(), String.t(), [String.t()], keyword()) :: map()
  defp build_body(provider, query, documents, opts) do
    %{
      "query" => query,
      "documents" => documents,
      "model" => opts[:model] || provider.model,
      "return_documents" => Keyword.get(opts, :return_documents, false)
    }
    |> maybe_put("top_n", opts[:top_n])
  end

  @spec validate_query(term()) :: :ok | {:error, Error.t()}
  defp validate_query(query) when is_binary(query) and query != "", do: :ok

  defp validate_query(query) do
    {:error,
     Error.new(
       :invalid_request,
       "query must be a non-empty string, got #{inspect(query)}",
       JinaRerankerM0
     )}
  end

  @spec validate_documents(term()) :: :ok | {:error, Error.t()}
  defp validate_documents([]) do
    {:error,
     Error.new(
       :invalid_request,
       "there is nothing to rerank: documents is empty",
       JinaRerankerM0
     )}
  end

  defp validate_documents(documents) when is_list(documents) do
    cond do
      length(documents) > @max_documents ->
        {:error,
         Error.new(
           :invalid_request,
           "at most #{@max_documents} documents per request, got #{length(documents)}; " <>
             "rerank a shortlist, not a corpus",
           JinaRerankerM0
         )}

      not Enum.all?(documents, &is_binary/1) ->
        {:error,
         Error.new(
           :invalid_request,
           "every document must be a string; render your records to text first",
           JinaRerankerM0
         )}

      true ->
        :ok
    end
  end

  defp validate_documents(documents) do
    {:error,
     Error.new(
       :invalid_request,
       "documents must be a list of strings, got #{inspect(documents)}",
       JinaRerankerM0
     )}
  end

  # The server rejects unknown body fields outright, so an unrecognized option is
  # a typo worth naming rather than something to forward and hope.
  @spec validate_opts(keyword()) :: :ok | {:error, Error.t()}
  defp validate_opts(opts) do
    case Keyword.keys(opts) -- @known_opts do
      [] ->
        :ok

      unknown ->
        {:error,
         Error.new(
           :invalid_request,
           "unknown option(s) #{inspect(unknown)}; this endpoint accepts #{inspect(@known_opts)}",
           JinaRerankerM0
         )}
    end
  end

  @spec validate_top_n(term()) :: :ok | {:error, Error.t()}
  defp validate_top_n(nil), do: :ok
  defp validate_top_n(top_n) when is_integer(top_n) and top_n > 0, do: :ok

  defp validate_top_n(top_n) do
    {:error,
     Error.new(
       :invalid_request,
       ":top_n must be a positive integer, got #{inspect(top_n)}",
       JinaRerankerM0
     )}
  end

  @spec build_result(map(), JinaRerankerM0.t(), non_neg_integer()) ::
          {:ok, Reranking.t()} | {:error, Error.t()}
  defp build_result(%{"results" => results} = body, provider, count) when is_list(results) do
    # A result with no numeric score, or an index outside the input, is not
    # usable: `nil >= threshold` is *true* in Elixir term ordering, so an
    # unscored entry would survive every relevance floor, and a bad index would
    # put `nil` into a prompt. Both are rejected here rather than downstream.
    if Enum.all?(results, &valid_result?(&1, count)) do
      {:ok,
       %Reranking{
         results: Enum.map(results, &normalize_result/1),
         model: body["model"] || provider.model,
         provider: JinaRerankerM0,
         # The server reports no token counts.
         usage: %{}
       }}
    else
      {:error, Error.unexpected_response(body, JinaRerankerM0)}
    end
  end

  defp build_result(body, _provider, _count),
    do: {:error, Error.unexpected_response(body, JinaRerankerM0)}

  @spec valid_result?(term(), non_neg_integer()) :: boolean()
  defp valid_result?(%{"index" => index, "relevance_score" => score}, count)
       when is_integer(index) and is_number(score),
       do: index >= 0 and index < count

  defp valid_result?(_result, _count), do: false

  @spec normalize_result(map()) :: Reranking.result()
  defp normalize_result(result) do
    %{
      index: result["index"],
      score: result["relevance_score"],
      document: document_text(result["document"])
    }
  end

  # `document` is an object, not a string - the multimodal shape leaves room for
  # an image alongside the text.
  @spec document_text(term()) :: String.t() | nil
  defp document_text(%{"text" => text}) when is_binary(text), do: text
  defp document_text(text) when is_binary(text), do: text
  defp document_text(_other), do: nil

  @spec maybe_put(map(), String.t(), term()) :: map()
  defp maybe_put(body, _key, nil), do: body
  defp maybe_put(body, key, value), do: Map.put(body, key, value)
end
