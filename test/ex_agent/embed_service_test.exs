defmodule ExAgent.EmbedServiceTest do
  @moduledoc """
  Provider-level embedding behaviour, driven through `ExAgent.embed/3`.
  """

  use ExUnit.Case, async: true

  alias ExAgent.{Embeddings, Error}
  alias ExAgent.Providers.{Gemini, JinaV5, OpenAI}
  alias ExAgent.Test.MinimalProvider

  # Records the request body, then answers with `vectors`.
  defp openai_style_provider(module, vectors, extra \\ []) do
    test_pid = self()

    plug = fn conn ->
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      send(test_pid, {:request, conn.request_path, Jason.decode!(body)})

      data =
        vectors
        |> Enum.with_index()
        |> Enum.map(fn {vector, index} -> %{"index" => index, "embedding" => vector} end)

      Req.Test.json(conn, %{"data" => data})
    end

    base =
      case module do
        OpenAI ->
          %OpenAI{api_key: "sk-test", model: "gpt-4o", base_url: "https://api.openai.com/v1"}

        JinaV5 ->
          %JinaV5{api_key: "jina-test", base_url: "https://jina.test"}
      end

    struct!(base, Keyword.merge([req: Req.new(plug: plug)], extra))
  end

  # The v5 server has its own body and response shape — `texts` in, `embeddings`
  # out, at /embed — so it needs its own stub rather than the OpenAI-style one.
  defp jina_provider(vectors) do
    test_pid = self()

    plug = fn conn ->
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      send(test_pid, {:request, conn.request_path, Jason.decode!(body)})

      Req.Test.json(conn, %{
        "model" => "jina-embeddings-v5-text-small",
        "input_count" => length(vectors),
        "embeddings" => vectors
      })
    end

    %JinaV5{api_key: "jina-test", base_url: "https://jina.test", req: Req.new(plug: plug)}
  end

  defp gemini_provider(vectors) do
    test_pid = self()

    plug = fn conn ->
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      send(test_pid, {:request, conn.request_path, Jason.decode!(body)})

      Req.Test.json(conn, %{"embeddings" => Enum.map(vectors, &%{"values" => &1})})
    end

    %Gemini{
      api_key: "AIza-test",
      model: "gemini-2.0-flash",
      base_url: "https://generativelanguage.googleapis.com/v1beta",
      req: Req.new(plug: plug)
    }
  end

  defp sent do
    assert_received {:request, path, body}
    %{path: path, body: body}
  end

  describe "ExAgent.embed/3 shape" do
    test "given three inputs, then three vectors come back from OpenAI" do
      provider = openai_style_provider(OpenAI, [[0.1], [0.2], [0.3]])

      assert {:ok, %Embeddings{} = result} =
               ExAgent.embed(provider, ["a", "b", "c"], task: nil)

      assert length(result.vectors) == 3
    end

    test "given a bare string, then it is wrapped into a single-element result" do
      provider = openai_style_provider(OpenAI, [[0.1, 0.2]])

      assert {:ok, %Embeddings{vectors: [[0.1, 0.2]]}} = ExAgent.embed(provider, "hello")
    end

    test "given a provider without embeddings, then it returns unsupported" do
      assert {:error, %Error{type: :unsupported} = error} =
               ExAgent.embed(MinimalProvider.new(), ["a"])

      assert error.message =~ "does not support embeddings"
      assert error.provider == MinimalProvider
    end
  end

  describe "OpenAI embeddings" do
    test "given no model, then it uses an embedding model, never the chat model" do
      provider = openai_style_provider(OpenAI, [[0.1]])

      assert {:ok, result} = ExAgent.embed(provider, ["a"])

      assert sent().body["model"] == "text-embedding-3-small"
      assert result.model == "text-embedding-3-small"
      refute result.model == provider.model
    end

    test "given an explicit model, then it is used" do
      provider = openai_style_provider(OpenAI, [[0.1]])

      assert {:ok, result} = ExAgent.embed(provider, ["a"], model: "text-embedding-3-large")

      assert sent().body["model"] == "text-embedding-3-large"
      assert result.model == "text-embedding-3-large"
    end

    test "given dimensions, then they are sent and recorded" do
      provider = openai_style_provider(OpenAI, [[0.1, 0.2, 0.3]])

      assert {:ok, result} = ExAgent.embed(provider, ["a"], dimensions: 768)

      assert sent().body["dimensions"] == 768
      assert result.dimensions == 768
    end

    test "given a task, then it errors rather than silently dropping it" do
      provider = openai_style_provider(OpenAI, [[0.1]])

      assert {:error, %Error{type: :unsupported} = error} =
               ExAgent.embed(provider, ["a"], task: :similarity)

      assert error.message =~ "no task type"
      refute_received {:request, _, _}
    end

    test "given out-of-order response data, then vectors are re-sorted by index" do
      test_pid = self()

      plug = fn conn ->
        send(test_pid, :called)

        Req.Test.json(conn, %{
          "data" => [
            %{"index" => 2, "embedding" => [0.3]},
            %{"index" => 0, "embedding" => [0.1]},
            %{"index" => 1, "embedding" => [0.2]}
          ]
        })
      end

      provider = %OpenAI{
        api_key: "sk-test",
        model: "gpt-4o",
        base_url: "https://api.openai.com/v1",
        req: Req.new(plug: plug)
      }

      assert {:ok, %Embeddings{vectors: vectors}} = ExAgent.embed(provider, ["a", "b", "c"])

      assert vectors == [[0.1], [0.2], [0.3]]
    end

    test "given a non-text input, then it is rejected" do
      provider = openai_style_provider(OpenAI, [[0.1]])

      assert {:error, %Error{type: :invalid_request}} =
               ExAgent.embed(provider, [%{content: "a", title: "t"}])
    end

    test "given a 429, then the normalized error surfaces" do
      plug = fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.send_resp(429, Jason.encode!(%{"error" => %{"message" => "slow down"}}))
      end

      provider = %OpenAI{
        api_key: "sk-test",
        model: "gpt-4o",
        base_url: "https://api.openai.com/v1",
        req: Req.new(plug: plug, retry: false)
      }

      assert {:error, %Error{type: :rate_limit, retryable?: true}} =
               ExAgent.embed(provider, ["a"])
    end
  end

  describe "Gemini embeddings — model families" do
    test "given the default model, then it uses gemini-embedding-001 via batchEmbedContents" do
      provider = gemini_provider([[0.1]])

      assert {:ok, result} = ExAgent.embed(provider, ["a"])

      assert %{path: path} = sent()
      assert path == "/models/gemini-embedding-001:batchEmbedContents"
      assert result.model == "gemini-embedding-001"
      # Never the chat model.
      refute result.model == provider.model
    end

    test "given 001 and a task, then the task becomes a taskType enum field" do
      provider = gemini_provider([[0.1]])

      assert {:ok, _} = ExAgent.embed(provider, ["a"], task: :retrieval_query)

      [request] = sent().body["requests"]
      assert request["taskType"] == "RETRIEVAL_QUERY"
      assert request["content"]["parts"] == [%{"text" => "a"}]
    end

    test "given 001, then every task maps to its documented enum" do
      expected = %{
        retrieval_query: "RETRIEVAL_QUERY",
        retrieval_document: "RETRIEVAL_DOCUMENT",
        similarity: "SEMANTIC_SIMILARITY",
        classification: "CLASSIFICATION",
        clustering: "CLUSTERING",
        question_answering: "QUESTION_ANSWERING",
        fact_verification: "FACT_VERIFICATION",
        code_query: "CODE_RETRIEVAL_QUERY"
      }

      for {task, enum} <- expected do
        provider = gemini_provider([[0.1]])
        assert {:ok, _} = ExAgent.embed(provider, ["a"], task: task)
        [request] = sent().body["requests"]
        assert request["taskType"] == enum, "#{task} should map to #{enum}"
      end
    end

    test "given gemini-embedding-2 and a query task, then the task becomes a text prefix" do
      provider = gemini_provider([[0.1]])

      assert {:ok, _} =
               ExAgent.embed(provider, ["what is otp?"],
                 model: "gemini-embedding-2",
                 task: :retrieval_query
               )

      [request] = sent().body["requests"]

      # No enum field on this family.
      refute Map.has_key?(request, "taskType")

      assert request["content"]["parts"] == [
               %{"text" => "task: search result | query: what is otp?"}
             ]
    end

    test "given gemini-embedding-2 and a document task, then title and text are templated" do
      provider = gemini_provider([[0.1]])

      assert {:ok, _} =
               ExAgent.embed(provider, [%{content: "body text", title: "OTP Guide"}],
                 model: "gemini-embedding-2",
                 task: :retrieval_document
               )

      [request] = sent().body["requests"]
      assert request["content"]["parts"] == [%{"text" => "title: OTP Guide | text: body text"}]
    end

    test "given a document with no title, then the template uses \"none\"" do
      provider = gemini_provider([[0.1]])

      assert {:ok, _} =
               ExAgent.embed(provider, ["body text"],
                 model: "gemini-embedding-2",
                 task: :retrieval_document
               )

      [request] = sent().body["requests"]
      assert request["content"]["parts"] == [%{"text" => "title: none | text: body text"}]
    end

    test "given an unknown embedding model, then it fails loudly instead of guessing" do
      provider = gemini_provider([[0.1]])

      assert {:error, %Error{type: :unsupported} = error} =
               ExAgent.embed(provider, ["a"], model: "gemini-embedding-99")

      assert error.message =~ "embedding_family"
      refute_received {:request, _, _}
    end

    test "given :embedding_family, then an unknown model is accepted" do
      provider = gemini_provider([[0.1]])

      assert {:ok, _} =
               ExAgent.embed(provider, ["a"],
                 model: "gemini-embedding-99",
                 task: :retrieval_query,
                 embedding_family: :prefix
               )

      [request] = sent().body["requests"]
      refute Map.has_key?(request, "taskType")
      assert request["content"]["parts"] == [%{"text" => "task: search result | query: a"}]
    end

    test "given an unknown task, then it is rejected before any request" do
      provider = gemini_provider([[0.1]])

      assert {:error, %Error{type: :invalid_request}} =
               ExAgent.embed(provider, ["a"], task: :summarization)

      refute_received {:request, _, _}
    end
  end

  describe "Gemini embeddings — the aggregation regression" do
    # A flat multi-input list returns ONE aggregated vector on gemini-embedding-2.
    # Wrapping each input in its own Content is what keeps an index from being
    # silently corrupted.
    test "given three inputs on gemini-embedding-2, then three vectors come back" do
      provider = gemini_provider([[0.1], [0.2], [0.3]])

      assert {:ok, %Embeddings{vectors: vectors}} =
               ExAgent.embed(provider, ["one", "two", "three"],
                 model: "gemini-embedding-2",
                 task: :retrieval_document
               )

      assert length(vectors) == 3
    end

    test "given three inputs, then the request carries three separate Content objects" do
      provider = gemini_provider([[0.1], [0.2], [0.3]])

      assert {:ok, _} =
               ExAgent.embed(provider, ["one", "two", "three"], model: "gemini-embedding-2")

      requests = sent().body["requests"]

      # Asserting on the request, not just the response: the failure mode is a
      # 200 carrying one aggregated vector, which a response-only assertion
      # detects but cannot diagnose.
      assert length(requests) == 3

      assert Enum.map(requests, & &1["content"]["parts"]) == [
               [%{"text" => "one"}],
               [%{"text" => "two"}],
               [%{"text" => "three"}]
             ]
    end

    test "given three inputs on 001, then three vectors come back too" do
      provider = gemini_provider([[0.1], [0.2], [0.3]])

      assert {:ok, %Embeddings{vectors: vectors}} =
               ExAgent.embed(provider, ["one", "two", "three"], task: :retrieval_document)

      assert length(vectors) == 3
      assert length(sent().body["requests"]) == 3
    end
  end

  describe "Gemini embeddings — dimensions and normalization" do
    test "given dimensions on 001, then outputDimensionality is sent" do
      provider = gemini_provider([[3.0, 4.0]])

      assert {:ok, _} = ExAgent.embed(provider, ["a"], dimensions: 768)

      [request] = sent().body["requests"]
      assert request["outputDimensionality"] == 768
    end

    test "given truncated dimensions on 001, then vectors are normalized client-side" do
      # 001 does not renormalize a truncated vector; cosine similarity needs it.
      provider = gemini_provider([[3.0, 4.0]])

      assert {:ok, %Embeddings{vectors: [vector]}} =
               ExAgent.embed(provider, ["a"], dimensions: 768)

      assert_in_delta magnitude(vector), 1.0, 1.0e-9
      assert vector == [0.6, 0.8]
    end

    test "given full dimensions on 001, then vectors are left untouched" do
      provider = gemini_provider([[3.0, 4.0]])

      assert {:ok, %Embeddings{vectors: [vector]}} =
               ExAgent.embed(provider, ["a"], dimensions: 3072)

      assert vector == [3.0, 4.0]
    end

    test "given no dimensions on 001, then vectors are left untouched" do
      provider = gemini_provider([[3.0, 4.0]])

      assert {:ok, %Embeddings{vectors: [vector]}} = ExAgent.embed(provider, ["a"])

      assert vector == [3.0, 4.0]
    end

    test "given truncated dimensions on gemini-embedding-2, then it self-normalizes" do
      provider = gemini_provider([[3.0, 4.0]])

      assert {:ok, %Embeddings{vectors: [vector]}} =
               ExAgent.embed(provider, ["a"], model: "gemini-embedding-2", dimensions: 768)

      # Not rescaled client-side — this family normalizes its own output.
      assert vector == [3.0, 4.0]
    end
  end

  # v5 splits what v3 fused: `task` says what kind of embedding, `prompt_name` says
  # which side of a retrieval pair. Both are v5's own vocabulary, so both are
  # validated here rather than translated from an invented shared set.
  describe "JinaV5 embeddings" do
    test "given a task, then v5's own spelling is sent to /embed" do
      provider = jina_provider([[0.1]])

      assert {:ok, _} = ExAgent.embed(provider, ["a"], task: :text_matching)

      sent = sent()
      assert sent.path == "/embed"
      assert sent.body["task"] == "text-matching"
      assert sent.body["texts"] == ["a"]
    end

    test "given a retrieval task and a prompt_name, then both reach the body" do
      provider = jina_provider([[0.1]])

      assert {:ok, _} =
               ExAgent.embed(provider, ["a"], task: :retrieval, args: [prompt_name: :document])

      body = sent().body
      assert body["task"] == "retrieval"
      assert body["prompt_name"] == "document"
    end

    # The server defaults to retrieval, which then demands a prompt_name; making
    # the default explicit is what lets that rule be checked before the request.
    test "given no task, then retrieval is assumed and a prompt_name is required" do
      provider = jina_provider([[0.1]])

      assert {:error, %Error{type: :invalid_request} = error} = ExAgent.embed(provider, ["a"])

      assert error.message =~ "task :retrieval needs args: [prompt_name:"
      refute_received {:request, _path, _body}
    end

    test "given a response, then the server's reported model is recorded" do
      provider = jina_provider([[0.1]])

      assert {:ok, result} = ExAgent.embed(provider, ["a"], task: :clustering)

      # The model is the server's to name; the body never carries one.
      refute Map.has_key?(sent().body, "model")
      assert result.model == "jina-embeddings-v5-text-small"
    end

    # The server truncates *and* re-normalizes, and honours `normalize: false`.
    # Re-normalizing client-side would silently undo that choice.
    test "given a Matryoshka dimension, then it is sent and the vectors left alone" do
      provider = jina_provider([[3.0, 4.0]])

      assert {:ok, result} =
               ExAgent.embed(provider, ["a"], task: :clustering, dimensions: 256)

      assert sent().body["dimensions"] == 256
      assert result.vectors == [[3.0, 4.0]]
      assert result.dimensions == 256
    end

    test "given more inputs than the server accepts, then the cap is named up front" do
      provider = jina_provider([[0.1]])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.embed(provider, Enum.map(1..513, &"chunk #{&1}"), task: :clustering)

      assert error.message =~ "at most 512 inputs"
      refute_received {:request, _path, _body}
    end

    test "given a prompt_name on a non-retrieval task, then it is refused" do
      provider = jina_provider([[0.1]])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.embed(provider, ["a"], task: :clustering, args: [prompt_name: :query])

      assert error.message =~ "applies only to task :retrieval"
      refute_received {:request, _path, _body}
    end

    test "given an untrained dimension, then it is rejected before the request" do
      provider = jina_provider([[0.1]])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.embed(provider, ["a"], task: :clustering, dimensions: 300)

      assert error.message =~ "trained for"
      refute_received {:request, _path, _body}
    end

    test "given a task from another provider's vocabulary, then it is rejected" do
      provider = jina_provider([[0.1]])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.embed(provider, ["a"], task: :retrieval_document)

      assert error.message =~ "unknown task :retrieval_document"
      assert error.message =~ ":text_matching"
      refute_received {:request, _path, _body}
    end

    test "given chat, then it says it is an embeddings model" do
      assert {:error, %Error{type: :unsupported} = error} =
               ExAgent.Provider.chat(jina_provider([]), [])

      assert error.message =~ "embeddings model"
    end

    test "given embedding_tasks/1, then v5's four tasks come back" do
      assert ExAgent.embedding_tasks(jina_provider([])) ==
               [:retrieval, :text_matching, :clustering, :classification]
    end
  end

  # A task string would be forwarded verbatim and accepted with a 200 by an
  # endpoint that does not recognize it, leaving quietly wrong vectors in an index.
  describe "task vocabularies are per provider" do
    test "given a string task on Jina, then it is rejected in favour of the atoms" do
      provider = jina_provider([[0.1]])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.embed(provider, ["a"], task: "text-matching")

      assert error.message =~ "must be an atom"
      assert error.message =~ ":text_matching"
    end

    test "given a string task on Gemini, then it is rejected and points to the atoms" do
      provider = gemini_provider([[0.1, 0.2]])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.embed(provider, ["a"], task: "NEW_ENUM_VALUE")

      assert error.message =~ "must be an atom"
      assert error.message =~ "retrieval_query"
    end

    test "given a Jina task on Gemini, then it is rejected" do
      provider = gemini_provider([[0.1, 0.2]])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.embed(provider, ["a"], task: :text_matching)

      assert error.message =~ "unknown task :text_matching"
    end

    test "given any task on OpenAI, then it is unsupported" do
      provider = openai_style_provider(OpenAI, [[0.1]])

      assert {:error, %Error{type: :unsupported}} =
               ExAgent.embed(provider, ["a"], task: :retrieval_query)
    end

    test "given embedding_tasks/1, then each provider reports its own set" do
      assert ExAgent.embedding_tasks(gemini_provider([])) |> Enum.sort() ==
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

      assert ExAgent.embedding_tasks(openai_style_provider(OpenAI, [])) == []
      assert ExAgent.embedding_tasks(MinimalProvider.new()) == []
    end
  end

  describe ":args passthrough" do
    test "given an allowed arg, then it is forwarded verbatim" do
      provider = jina_provider([[0.1]])

      assert {:ok, _} =
               ExAgent.embed(provider, ["a"], task: :clustering, args: [normalize: false])

      assert sent().body["normalize"] == false
    end

    test "given args as a map, then it works the same" do
      provider = jina_provider([[0.1]])

      assert {:ok, _} = ExAgent.embed(provider, ["a"], args: %{prompt_name: "query"})

      assert sent().body["prompt_name"] == "query"
    end

    test "given an atom prompt_name, then it reaches the wire as a string" do
      provider = jina_provider([[0.1]])

      assert {:ok, _} =
               ExAgent.embed(provider, ["a"], task: :retrieval, args: [prompt_name: :document])

      assert sent().body["prompt_name"] == "document"
    end

    test "given an unknown arg, then the error lists what the provider accepts" do
      provider = jina_provider([[0.1]])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.embed(provider, ["a"], task: :clustering, args: [prompt_nane: "document"])

      assert error.message =~ "does not accept :prompt_nane"
      assert error.message =~ ":prompt_name"
      refute_received {:request, _path, _body}
    end

    test "given an invalid value for a known arg, then it names the allowed values" do
      provider = jina_provider([[0.1]])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.embed(provider, ["a"], task: :retrieval, args: [prompt_name: :passage])

      assert error.message =~ "invalid value :passage"
      assert error.message =~ "\"document\""
    end

    test "given args on a provider that accepts none, then it says so" do
      provider = gemini_provider([[0.1, 0.2]])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.embed(provider, ["a"], args: [title: "doc"])

      assert error.message =~ "accepts no extra :args"
    end

    test "given OpenAI's own args, then they are forwarded" do
      provider = openai_style_provider(OpenAI, [[0.1]])

      assert {:ok, _} = ExAgent.embed(provider, ["a"], args: [encoding_format: "float"])

      assert sent().body["encoding_format"] == "float"
    end

    test "given args in the wrong shape, then it is rejected" do
      provider = jina_provider([[0.1]])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.embed(provider, ["a"], task: :clustering, args: "prompt_name=query")

      assert error.message =~ "keyword list or a map"
    end
  end

  describe "result provenance" do
    test "given a result, then model, dimensions and task are carried for storage" do
      provider = gemini_provider([[0.1, 0.2, 0.3]])

      assert {:ok, result} =
               ExAgent.embed(provider, ["a"], task: :retrieval_document, dimensions: 3)

      assert result.model == "gemini-embedding-001"
      assert result.provider == Gemini
      assert result.dimensions == 3
      assert result.task == :retrieval_document
    end

    test "given a response with usage, then token counts are carried" do
      plug = fn conn ->
        Req.Test.json(conn, %{
          "data" => [%{"index" => 0, "embedding" => [0.1]}],
          "usage" => %{"prompt_tokens" => 7, "total_tokens" => 7}
        })
      end

      provider = %OpenAI{
        api_key: "sk-test",
        model: "gpt-4o",
        base_url: "https://api.openai.com/v1",
        req: Req.new(plug: plug)
      }

      assert {:ok, result} = ExAgent.embed(provider, ["a"])
      assert result.usage == %{input_tokens: 7, total_tokens: 7}
    end

    test "given a response without usage, then it stays empty rather than nil" do
      provider = openai_style_provider(OpenAI, [[0.1]])

      assert {:ok, %Embeddings{usage: %{}}} = ExAgent.embed(provider, ["a"])
    end

    test "given no explicit dimensions, then they are inferred from the vectors" do
      provider = gemini_provider([[0.1, 0.2, 0.3, 0.4]])

      assert {:ok, result} = ExAgent.embed(provider, ["a"])

      assert result.dimensions == 4
    end
  end

  defp magnitude(vector) do
    vector |> Enum.reduce(0.0, fn v, acc -> acc + v * v end) |> :math.sqrt()
  end
end
