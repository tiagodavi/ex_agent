defmodule ExAgent.EmbeddingsTest do
  use ExUnit.Case, async: true

  doctest ExAgent.Embeddings

  alias ExAgent.Embeddings

  describe "tasks/0 and valid_task?/1" do
    test "given the vocabulary, then it is exactly the eight normalized tasks" do
      assert Enum.sort(Embeddings.tasks()) ==
               Enum.sort([
                 :retrieval_query,
                 :retrieval_document,
                 :similarity,
                 :classification,
                 :clustering,
                 :question_answering,
                 :fact_verification,
                 :code_query
               ])
    end

    test "given a known task, then it validates" do
      for task <- Embeddings.tasks(), do: assert(Embeddings.valid_task?(task))
    end

    test "given an unknown task, then it does not validate" do
      refute Embeddings.valid_task?(:summarization)
      refute Embeddings.valid_task?("retrieval_query")
      refute Embeddings.valid_task?(nil)
    end
  end

  describe "l2_normalize/1" do
    test "given a vector, then the result has unit magnitude" do
      normalized = Embeddings.l2_normalize([1.0, 2.0, 3.0, 4.0])

      assert_in_delta magnitude(normalized), 1.0, 1.0e-9
    end

    test "given an already-unit vector, then it is unchanged" do
      assert_in_delta magnitude(Embeddings.l2_normalize([1.0, 0.0, 0.0])), 1.0, 1.0e-9
    end

    test "given a zero vector, then it is returned without dividing by zero" do
      assert Embeddings.l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
    end

    test "given negative components, then direction is preserved" do
      [a, b] = Embeddings.l2_normalize([-3.0, 4.0])

      assert a < 0
      assert b > 0
      assert_in_delta magnitude([a, b]), 1.0, 1.0e-9
    end
  end

  describe "cosine_similarity/2" do
    test "given identical vectors, then similarity is 1.0" do
      assert_in_delta Embeddings.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, 1.0e-9
    end

    test "given orthogonal vectors, then similarity is 0.0" do
      assert_in_delta Embeddings.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, 1.0e-9
    end

    test "given opposite vectors, then similarity is -1.0" do
      assert_in_delta Embeddings.cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0, 1.0e-9
    end

    test "given a zero vector, then similarity is 0.0 rather than a division error" do
      assert Embeddings.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
      assert Embeddings.cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0
    end

    test "given vectors of different lengths, then it raises rather than comparing" do
      assert_raise ArgumentError, ~r/different lengths/, fn ->
        Embeddings.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
      end
    end

    test "given magnitude differences only, then similarity ignores scale" do
      assert_in_delta Embeddings.cosine_similarity([1.0, 1.0], [5.0, 5.0]), 1.0, 1.0e-9
    end
  end

  defp magnitude(vector) do
    vector |> Enum.reduce(0.0, fn v, acc -> acc + v * v end) |> :math.sqrt()
  end
end
