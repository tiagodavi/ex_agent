defmodule ExAgent.Reranking do
  @moduledoc """
  Result of a rerank request: documents reordered by relevance to a query.

      {:ok, result} = ExAgent.rerank(provider, "who restarts processes?", chunks)

      for %{index: index, score: score} <- result.results do
        IO.puts("\#{score} - \#{Enum.at(chunks, index)}")
      end

  ## Why rerank at all

  Embedding retrieval compares a query vector to document vectors computed
  *independently*, which is what makes it fast enough to run over millions of
  rows - and also what limits it. A cross-encoder reads the query and one
  document **together**, so it can weigh them against each other, at a cost that
  only makes sense over a shortlist.

  The pairing is therefore: embeddings fetch the top ~100, a reranker orders the
  top ~10 that actually go in the prompt.

  ## `:index` is the contract, not `:document`

  Every entry carries the **position of the document in the list you passed**, so
  results map back to your own records without depending on the server echoing
  text back:

      ranked = Enum.map(result.results, fn %{index: i} -> Enum.at(rows, i) end)

  `:document` is populated only when the server returns the text, which is
  optional and off by default - sending a corpus back over the wire to get it
  re-quoted is waste.

  ## Scores are model-scoped

  Higher means more relevant, and that is the only guarantee. The range is the
  model's own: `jina-reranker-m0` emits roughly 0-1 while other cross-encoders
  emit unbounded logits. So compare scores **within one result set**, never
  across models, and be wary of absolute cutoffs - persist `:model` next to
  anything you store, for the same reason `ExAgent.Embeddings` does.
  """

  @typedoc """
  One reranked document.

  `:index` is its position in the input list; `:score` is the model's relevance
  score; `:document` is the text, present only when the provider returned it.
  """
  @type result :: %{
          required(:index) => non_neg_integer(),
          required(:score) => float(),
          optional(:document) => String.t() | nil
        }

  @type t :: %__MODULE__{
          results: [result()],
          model: String.t(),
          provider: module(),
          usage: map()
        }

  @enforce_keys [:results, :model, :provider]
  defstruct [:results, :model, :provider, usage: %{}]

  @doc """
  Reorders `documents` to match the ranking, dropping anything not ranked.

  A convenience for the common case where the ranked *texts* are all you need.
  Use `:index` directly when you are ranking rows rather than strings.

  ## Examples

      iex> result = %ExAgent.Reranking{
      ...>   results: [%{index: 2, score: 0.9}, %{index: 0, score: 0.4}],
      ...>   model: "m",
      ...>   provider: Fake
      ...> }
      iex> ExAgent.Reranking.take(result, ["a", "b", "c"])
      ["c", "a"]
  """
  @spec take(t(), [term()]) :: [term()]
  def take(%__MODULE__{results: results}, documents) do
    count = length(documents)

    Enum.map(results, fn %{index: index} ->
      # Silently returning nil here would put "nil" in a prompt. An out-of-range
      # index means this is not the list that was ranked.
      if index < 0 or index >= count do
        raise ArgumentError,
              "result index #{index} is outside the #{count} documents given; " <>
                "take/2 needs the same list that was ranked"
      end

      Enum.at(documents, index)
    end)
  end

  @doc """
  Keeps only results scoring at or above `threshold`.

  Ranking always returns *something*, even when nothing is relevant - the best of
  a bad set still sorts first. A floor is how you decline to answer instead of
  feeding the model the least-irrelevant chunk.

  The threshold is model-specific, so calibrate it against your own data rather
  than copying a number.

  ## Examples

      iex> result = %ExAgent.Reranking{
      ...>   results: [%{index: 0, score: 0.9}, %{index: 1, score: 0.1}],
      ...>   model: "m",
      ...>   provider: Fake
      ...> }
      iex> ExAgent.Reranking.above(result, 0.5).results
      [%{index: 0, score: 0.9}]
  """
  @spec above(t(), number()) :: t()
  def above(%__MODULE__{results: results} = reranking, threshold) when is_number(threshold) do
    # `is_number/1` first: in Elixir's term ordering every atom sorts above every
    # number, so a `nil` score would pass `>= threshold` and survive the filter.
    %{reranking | results: Enum.filter(results, &(is_number(&1.score) and &1.score >= threshold))}
  end
end
