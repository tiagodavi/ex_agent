defmodule ExAgent.Embeddings do
  @moduledoc """
  Result of an embedding request, with the provenance needed to use it safely.

      {:ok, result} = ExAgent.embed(provider, ["hello", "world"], task: :retrieval_document)
      result.vectors  #=> [[0.01, ...], [0.02, ...]]

  ## Store the provenance alongside every vector

  Embedding spaces are model-scoped. Vectors from `gemini-embedding-001` and
  `gemini-embedding-2` are **not** comparable — mixing them silently degrades
  retrieval rather than failing, and the only fix is a full re-embed. The same
  applies to a change in `:dimensions` or `:task`.

  Persist `:model`, `:dimensions`, and `:task` next to the vector itself, so a
  later model change is a detectable migration rather than a quiet regression.

  ## Tasks

  Providers express "what is this embedding for" in incompatible ways — an enum
  field, a text prefix, or a body parameter. `ExAgent.embed/3` takes one
  normalized vocabulary and translates:

  | Task | Use |
  |------|-----|
  | `:retrieval_query` | The search query side of retrieval |
  | `:retrieval_document` | The indexed-document side of retrieval |
  | `:similarity` | Symmetric semantic similarity |
  | `:classification` | Features for a classifier |
  | `:clustering` | Features for clustering |
  | `:question_answering` | Question side of QA retrieval |
  | `:fact_verification` | Claim side of fact checking |
  | `:code_query` | Natural-language query over code |

  `:retrieval_query` and `:retrieval_document` are deliberately distinct.
  Asymmetric retrieval needs both, and using one where the other belongs
  degrades recall invisibly.
  """

  @type task ::
          :retrieval_query
          | :retrieval_document
          | :similarity
          | :classification
          | :clustering
          | :question_answering
          | :fact_verification
          | :code_query

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

  @tasks ~w(retrieval_query retrieval_document similarity classification
            clustering question_answering fact_verification code_query)a

  @doc """
  Returns the normalized task vocabulary.

  ## Examples

      iex> :retrieval_query in ExAgent.Embeddings.tasks()
      true
  """
  @spec tasks() :: [task()]
  def tasks, do: @tasks

  @doc """
  Returns whether `task` is part of the normalized vocabulary.

  ## Examples

      iex> ExAgent.Embeddings.valid_task?(:clustering)
      true

      iex> ExAgent.Embeddings.valid_task?(:summarization)
      false
  """
  @spec valid_task?(term()) :: boolean()
  def valid_task?(task), do: task in @tasks

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
  length — that almost always means they came from different models or
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
