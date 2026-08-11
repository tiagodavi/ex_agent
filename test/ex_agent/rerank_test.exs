defmodule ExAgent.RerankTest do
  @moduledoc """
  Reranking, driven through `ExAgent.rerank/4`.

  The wire shape asserted here was taken from a running `jina-reranker-m0`
  deployment: `relevance_score` rather than `score`, `document` nested under a
  `"text"` key, and results already ordered by the server.
  """

  use ExUnit.Case, async: true

  doctest ExAgent.Reranking

  alias ExAgent.{Error, Reranking}
  alias ExAgent.Providers.{JinaRerankerM0, OpenAI}
  alias ExAgent.Test.MinimalProvider

  @docs ["supervisors restart children", "unrelated text", "beam bytecode"]

  # Answers with the server's real response shape, recording the request.
  defp reranker(results, opts \\ []) do
    test_pid = self()

    plug = fn conn ->
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      send(test_pid, {:request, conn.request_path, Jason.decode!(body)})

      Req.Test.json(conn, %{
        "model" => Keyword.get(opts, :model, "jinaai/jina-reranker-m0"),
        "results" => results
      })
    end

    %JinaRerankerM0{
      base_url: "https://rerank.test",
      api_key: "test",
      model: "jina-reranker-m0",
      req: Req.new(plug: plug)
    }
  end

  defp scored(pairs) do
    Enum.map(pairs, fn {index, score} -> %{"index" => index, "relevance_score" => score} end)
  end

  defp sent do
    assert_received {:request, path, body}
    %{path: path, body: body}
  end

  describe "rerank/4" do
    test "given documents, then they come back ordered with scores and indexes" do
      provider = reranker(scored([{0, 0.9}, {2, 0.3}, {1, 0.05}]))

      assert {:ok, %Reranking{} = result} = ExAgent.rerank(provider, "supervision", @docs)

      assert result.results == [
               %{index: 0, score: 0.9, document: nil},
               %{index: 2, score: 0.3, document: nil},
               %{index: 1, score: 0.05, document: nil}
             ]

      assert result.model == "jinaai/jina-reranker-m0"
      assert result.provider == JinaRerankerM0
    end

    test "given a request, then it goes to /v1/rerank with query and documents" do
      provider = reranker(scored([{0, 0.5}]))

      assert {:ok, _result} = ExAgent.rerank(provider, "supervision", @docs)

      sent = sent()
      assert sent.path == "/v1/rerank"
      assert sent.body["query"] == "supervision"
      assert sent.body["documents"] == @docs
      assert sent.body["model"] == "jina-reranker-m0"
    end

    test "given :top_n, then it is forwarded" do
      provider = reranker(scored([{0, 0.5}]))

      assert {:ok, _result} = ExAgent.rerank(provider, "q", @docs, top_n: 1)

      assert sent().body["top_n"] == 1
    end

    # `:index` is the contract, so echoing the corpus back is pure waste - but the
    # server defaults to doing it, which makes the explicit `false` load-bearing.
    test "given no option, then documents are not requested back" do
      provider = reranker(scored([{0, 0.5}]))

      assert {:ok, _result} = ExAgent.rerank(provider, "q", @docs)

      assert sent().body["return_documents"] == false
    end

    test "given :return_documents, then the text is carried on each result" do
      results = [
        %{"index" => 0, "relevance_score" => 0.9, "document" => %{"text" => "supervisors"}}
      ]

      provider = reranker(results)

      assert {:ok, result} = ExAgent.rerank(provider, "q", @docs, return_documents: true)

      assert [%{document: "supervisors"}] = result.results
      assert sent().body["return_documents"] == true
    end

    test "given a per-call :model, then it overrides the provider's" do
      provider = reranker(scored([{0, 0.5}]))

      assert {:ok, _result} = ExAgent.rerank(provider, "q", @docs, model: "other")

      assert sent().body["model"] == "other"
    end
  end

  describe "rerank/4 validation" do
    test "given no documents, then it says there is nothing to rerank" do
      provider = reranker([])

      assert {:error, %Error{type: :invalid_request} = error} = ExAgent.rerank(provider, "q", [])

      assert error.message =~ "nothing to rerank"
      refute_received {:request, _path, _body}
    end

    test "given more documents than the server accepts, then the cap is named" do
      provider = reranker([])
      documents = Enum.map(1..513, &"doc #{&1}")

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.rerank(provider, "q", documents)

      assert error.message =~ "at most 512 documents"
      assert error.message =~ "shortlist, not a corpus"
      refute_received {:request, _path, _body}
    end

    test "given non-string documents, then it says to render them first" do
      provider = reranker([])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.rerank(provider, "q", [%{text: "a record"}])

      assert error.message =~ "must be a string"
      refute_received {:request, _path, _body}
    end

    test "given an empty query, then it is rejected" do
      provider = reranker([])

      assert {:error, %Error{type: :invalid_request}} = ExAgent.rerank(provider, "", @docs)

      refute_received {:request, _path, _body}
    end

    # The server rejects unknown body fields, so a typo has to be caught here or
    # it becomes a 422 blob.
    test "given an unknown option, then it names the accepted ones" do
      provider = reranker([])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.rerank(provider, "q", @docs, topn: 3)

      assert error.message =~ "unknown option(s) [:topn]"
      assert error.message =~ ":top_n"
      refute_received {:request, _path, _body}
    end

    test "given a non-positive :top_n, then it is rejected" do
      provider = reranker([])

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.rerank(provider, "q", @docs, top_n: 0)

      assert error.message =~ "positive integer"
    end

    test "given an unparseable response, then it is a normalized error" do
      provider = reranker([])

      broken = %{
        provider
        | req: Req.new(plug: fn conn -> Req.Test.json(conn, %{"oops" => 1}) end)
      }

      assert {:error, %Error{type: :server}} = ExAgent.rerank(broken, "q", @docs)
    end
  end

  describe "providers without reranking" do
    test "given a chat provider, then reranking is unsupported" do
      provider = OpenAI.new(api_key: "sk-test")

      assert {:error, %Error{type: :unsupported} = error} = ExAgent.rerank(provider, "q", @docs)

      assert error.message =~ "does not support reranking"
    end

    test "given a minimal provider, then reranking is unsupported" do
      assert {:error, %Error{type: :unsupported}} =
               ExAgent.rerank(MinimalProvider.new(), "q", @docs)
    end

    test "given the reranker, then chat is unsupported and points elsewhere" do
      {:ok, message} = ExAgent.Message.new(role: :user, content: "hi")

      assert {:error, %Error{type: :unsupported} = error} =
               ExAgent.Provider.chat(reranker([]), [message])

      assert error.message =~ "reranking model"
    end
  end

  describe "Reranking helpers" do
    test "given a ranking, then take/2 reorders the original list" do
      result = %Reranking{
        results: [%{index: 2, score: 0.9}, %{index: 0, score: 0.4}],
        model: "m",
        provider: JinaRerankerM0
      }

      assert Reranking.take(result, @docs) == ["beam bytecode", "supervisors restart children"]
    end

    # Ranking always returns something: the best of an irrelevant set still sorts
    # first, so a floor is how you decline to answer.
    test "given a threshold, then above/2 drops what does not clear it" do
      result = %Reranking{
        results: [%{index: 0, score: 0.9}, %{index: 1, score: 0.2}],
        model: "m",
        provider: JinaRerankerM0
      }

      assert Reranking.above(result, 0.5).results == [%{index: 0, score: 0.9}]
      assert Reranking.above(result, 0.95).results == []
    end
  end

  describe "credentials" do
    test "given a reranker struct, then inspecting it redacts the key and headers" do
      provider =
        JinaRerankerM0.new(
          base_url: "https://rerank.test",
          api_key: "secret-key",
          headers: [{"Modal-Key", "wk-secret"}]
        )

      dumped = inspect(provider, limit: :infinity)

      refute dumped =~ "secret-key"
      refute dumped =~ "wk-secret"
    end
  end

  describe "telemetry" do
    test "given a rerank call, then a span reports the document count" do
      :telemetry.attach(
        "rerank-span",
        [:ex_agent, :rerank, :stop],
        fn _event, measurements, metadata, pid -> send(pid, {:span, measurements, metadata}) end,
        self()
      )

      on_exit(fn -> :telemetry.detach("rerank-span") end)

      assert {:ok, _result} = ExAgent.rerank(reranker(scored([{0, 0.5}])), "q", @docs)

      assert_received {:span, measurements, metadata}
      assert measurements.duration > 0
      assert metadata.document_count == 3
      assert metadata.provider == JinaRerankerM0
      assert metadata.result == :ok
    end
  end
end
