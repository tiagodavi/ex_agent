defmodule ExAgent.Embeddings do
  @moduledoc """
  Result of an embedding request, with the provenance needed to use it safely.

      {:ok, result} = ExAgent.embed(provider, ["hello", "world"], task: :retrieval_document)
      result.vectors  #=> [[0.01, ...], [0.02, ...]]

  ## Store the provenance alongside every vector

  Embedding spaces are model-scoped. Vectors from `gemini-embedding-001` and
  `gemini-embedding-2` are **not** comparable - mixing them silently degrades
  retrieval rather than failing, and the only fix is a full re-embed. The same
  applies to a change in `:dimensions` or `:task`.

  Persist `:model`, `:dimensions`, and `:task` next to the vector itself, so a
  later model change is a detectable migration rather than a quiet regression.

  ## Tasks belong to the provider

  "What is this embedding for" is asked in incompatible ways, and the vocabulary
  belongs to the **model**, not to this library: Gemini's `taskType` is a closed
  enum of eight values, Jina v5 takes four task names plus a separate
  `prompt_name`, and OpenAI has no task field at all. There is no portable middle
  - a shared vocabulary either loses the distinctions a model actually makes or
  invents ones it does not.

  So each provider declares its own atoms and rejects anything outside them:

      ExAgent.embedding_tasks(gemini)   #=> [:retrieval_query, :retrieval_document, ...]
      ExAgent.embedding_tasks(jina)     #=> [:retrieval, :text_matching, :clustering, :classification]

  A task the provider does not know is an error naming the ones it does, never a
  field quietly dropped: a dropped task returns HTTP 200 with plausible floats
  that land in a vector store and degrade retrieval forever, with no later signal.

  ## Extra model arguments

  `:args` forwards key-value pairs into the request body for parameters this
  library does not model - Jina's `prompt_name`, OpenAI's `encoding_format`:

      ExAgent.embed(jina, chunks, task: :retrieval, args: [prompt_name: :document])

  Each provider validates them against what its own endpoint accepts and rejects
  unknown keys, so a typo fails loudly instead of being ignored by the server.
  """

  @typedoc """
  What to pass as `:task`: an atom from the provider's own vocabulary.

  Discover it with `ExAgent.embedding_tasks/1`. There is deliberately no shared
  vocabulary and no verbatim-string escape hatch - a task string an endpoint does
  not recognize is accepted with a 200 and quietly wrong vectors.
  """
  @type task :: atom()

  @typedoc """
  Extra request-body parameters, validated by the provider before being sent.
  """
  @type args :: keyword() | map()

  @type input :: String.t() | %{optional(:content) => String.t(), optional(:title) => String.t()}

  @type t :: %__MODULE__{
          vectors: [[float()]],
          model: String.t(),
          provider: module(),
          dimensions: pos_integer() | nil,
          task: task() | nil,
          usage: map()
        }

  @enforce_keys [:vectors, :model, :provider]
  defstruct [:vectors, :model, :provider, :dimensions, :task, usage: %{}]

  @doc """
  Normalizes `:args` into a keyword list, rejecting anything else.

  Providers call this before validating against their own allowlist, so `:args`
  can be given as either a keyword list or a map.
  """
  @spec normalize_args(term()) :: {:ok, keyword()} | :error
  def normalize_args(nil), do: {:ok, []}
  def normalize_args(args) when is_map(args), do: {:ok, Map.to_list(args)}

  def normalize_args(args) when is_list(args) do
    if Keyword.keyword?(args), do: {:ok, args}, else: :error
  end

  def normalize_args(_args), do: :error

  @doc """
  Scales a vector to unit length.

  Some providers return non-unit vectors when the output is truncated to fewer
  dimensions; cosine similarity assumes unit length, so those must be normalized
  before use. A zero vector is returned unchanged rather than dividing by zero.

  ## Examples

      iex> ExAgent.Embeddings.l2_normalize([3.0, 4.0])
      [0.6, 0.8]

      iex> ExAgent.Embeddings.l2_normalize([0.0, 0.0])
      [0.0, 0.0]
  """
  @spec l2_normalize([float()]) :: [float()]
  def l2_normalize(vector) do
    magnitude = magnitude(vector)

    if magnitude == 0, do: vector, else: Enum.map(vector, &(&1 / magnitude))
  end

  @doc """
  Cosine similarity between two vectors, in `-1.0..1.0`.

  Returns `0.0` if either vector is all zeros. Raises when the vectors differ in
  length - that almost always means they came from different models or
  dimension settings, which is a bug rather than a value worth computing.

  ## Examples

      iex> ExAgent.Embeddings.cosine_similarity([1.0, 0.0], [1.0, 0.0])
      1.0

      iex> ExAgent.Embeddings.cosine_similarity([1.0, 0.0], [0.0, 1.0])
      0.0
  """
  @spec cosine_similarity([float()], [float()]) :: float()
  def cosine_similarity(a, b) when length(a) == length(b) do
    magnitude_a = magnitude(a)
    magnitude_b = magnitude(b)

    if magnitude_a == 0 or magnitude_b == 0 do
      0.0
    else
      dot(a, b) / (magnitude_a * magnitude_b)
    end
  end

  def cosine_similarity(a, b) do
    raise ArgumentError,
          "cannot compare vectors of different lengths (#{length(a)} and #{length(b)}); " <>
            "they are probably from different models or :dimensions settings"
  end

  @spec magnitude([float()]) :: float()
  defp magnitude(vector) do
    vector |> Enum.reduce(0.0, fn value, acc -> acc + value * value end) |> :math.sqrt()
  end

  @spec dot([float()], [float()]) :: float()
  defp dot(a, b) do
    a |> Enum.zip(b) |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
  end
end
