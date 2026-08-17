defmodule ExAgent.Services.ReceiveTimeoutTest do
  @moduledoc """
  `:receive_timeout` across every HTTP-facing service.

  The invariant under test is that a call-site timeout actually reaches Req.
  It used to be filtered out by each chat service's options schema, so a caller
  wrapping the request in a shorter supervising timeout - a `Task` killed at
  180 s around a request still willing to wait 300 s - had no way to make the
  HTTP layer give up first, and a slow-but-valid answer was reported as a
  dropped call.
  """

  use ExUnit.Case, async: true

  alias ExAgent.Message
  alias ExAgent.Providers.{Gemini, JinaRerankerM0, JinaV5, OpenAI, OpenAICompatible}

  alias ExAgent.Services.{
    GeminiEmbedService,
    GeminiService,
    GeminiUploadService,
    JinaRerankerService,
    JinaV5EmbedService,
    OpenAICompatibleService,
    OpenAIEmbedService,
    OpenAIService,
    OpenAIUploadService
  }

  @five_minutes :timer.minutes(5)
  @short :timer.seconds(150)

  # Records the timeouts Req was actually handed, after option merging.
  defp capturing_req(response) do
    test_pid = self()

    Req.new(plug: fn conn -> Req.Test.json(conn, response) end)
    |> Req.Request.append_request_steps(
      capture_timeouts: fn request ->
        send(
          test_pid,
          {:timeouts, request.options[:receive_timeout],
           get_in(request.options, [:connect_options, :timeout])}
        )

        request
      end
    )
  end

  defp captured do
    assert_received {:timeouts, receive_timeout, connect_timeout}
    %{receive: receive_timeout, connect: connect_timeout}
  end

  defp messages do
    {:ok, message} = Message.new(role: :user, content: "Hi")
    [message]
  end

  defp chat_reply,
    do: %{"choices" => [%{"message" => %{"role" => "assistant", "content" => "ok"}}]}

  defp gemini_reply, do: %{"candidates" => [%{"content" => %{"parts" => [%{"text" => "ok"}]}}]}

  defp openai(response, opts \\ []) do
    %OpenAI{
      api_key: "sk-test",
      model: "gpt-4o",
      base_url: "https://api.openai.com/v1",
      req: capturing_req(response)
    }
    |> struct!(opts)
  end

  defp compatible(response, opts \\ []) do
    %OpenAICompatible{
      base_url: "https://vllm.test/v1",
      model: "Qwen3-VL",
      modalities: [:text, :image, :video],
      req: capturing_req(response)
    }
    |> struct!(opts)
  end

  defp gemini(response, opts \\ []) do
    %Gemini{
      api_key: "AIza-test",
      model: "gemini-2.0-flash",
      base_url: "https://generativelanguage.googleapis.com/v1beta",
      req: capturing_req(response)
    }
    |> struct!(opts)
  end

  describe "provider defaults" do
    test "given a provider built with new/1, then the default is five minutes" do
      assert OpenAI.new(api_key: "sk-test").receive_timeout == @five_minutes
      assert Gemini.new(api_key: "AIza-test").receive_timeout == @five_minutes
      assert JinaV5.new(base_url: "https://jina.test").receive_timeout == @five_minutes
      assert JinaRerankerM0.new(base_url: "https://rr.test").receive_timeout == @five_minutes

      assert OpenAICompatible.new(base_url: "https://vllm.test/v1", model: "m").receive_timeout ==
               @five_minutes
    end

    test "given a non-positive timeout, then new/1 refuses it rather than sending it" do
      assert_raise NimbleOptions.ValidationError, fn ->
        OpenAI.new(api_key: "sk-test", receive_timeout: 0)
      end

      assert_raise NimbleOptions.ValidationError, fn ->
        OpenAICompatible.new(base_url: "https://vllm.test/v1", model: "m", receive_timeout: -1)
      end
    end
  end

  describe "chat" do
    test "given nothing set, then five minutes reaches Req on all three services" do
      assert {:ok, _} = OpenAICompatibleService.chat(compatible(chat_reply()), messages())
      assert %{receive: @five_minutes, connect: @five_minutes} = captured()

      assert {:ok, _} = OpenAIService.chat(openai(chat_reply()), messages())
      assert %{receive: @five_minutes, connect: @five_minutes} = captured()

      assert {:ok, _} = GeminiService.chat(gemini(gemini_reply()), messages())
      assert %{receive: @five_minutes, connect: @five_minutes} = captured()
    end

    test "given the provider field, then it replaces the default" do
      provider = compatible(chat_reply(), receive_timeout: @short)

      assert {:ok, _} = OpenAICompatibleService.chat(provider, messages())
      assert %{receive: @short, connect: @short} = captured()
    end

    test "given a call-site timeout, then it wins over the provider field" do
      opts = [receive_timeout: @short]

      assert {:ok, _} =
               OpenAICompatibleService.chat(
                 compatible(chat_reply(), receive_timeout: :timer.minutes(9)),
                 messages(),
                 opts
               )

      assert %{receive: @short, connect: @short} = captured()

      assert {:ok, _} = OpenAIService.chat(openai(chat_reply()), messages(), opts)
      assert %{receive: @short} = captured()

      assert {:ok, _} = GeminiService.chat(gemini(gemini_reply()), messages(), opts)
      assert %{receive: @short} = captured()
    end

    test "given a call-site timeout, then the connect timeout follows it too" do
      # A connect stuck at the old hardcoded five minutes would outlive a
      # shorter supervising timeout even when the receive timeout is lowered.
      assert {:ok, _} =
               OpenAIService.chat(openai(chat_reply()), messages(), receive_timeout: @short)

      assert %{connect: @short} = captured()
    end
  end

  describe "embeddings" do
    test "given no timeout, then the provider default reaches Req" do
      provider = openai(%{"data" => [%{"index" => 0, "embedding" => [0.1]}]})

      assert {:ok, _} = OpenAIEmbedService.embed(provider, ["a"])
      assert %{receive: @five_minutes} = captured()
    end

    test "given a call-site timeout, then it reaches Req on every embed service" do
      opts = [receive_timeout: @short]

      openai_provider = openai(%{"data" => [%{"index" => 0, "embedding" => [0.1]}]})
      assert {:ok, _} = OpenAIEmbedService.embed(openai_provider, ["a"], opts)
      assert %{receive: @short} = captured()

      gemini_provider = gemini(%{"embeddings" => [%{"values" => [0.1]}]})
      assert {:ok, _} = GeminiEmbedService.embed(gemini_provider, ["a"], opts)
      assert %{receive: @short} = captured()

      jina_provider = %JinaV5{
        base_url: "https://jina.test",
        api_key: "test",
        req: capturing_req(%{"embeddings" => [[0.1]]})
      }

      jina_opts = Keyword.put(opts, :args, prompt_name: :query)
      assert {:ok, _} = JinaV5EmbedService.embed(jina_provider, ["a"], jina_opts)
      assert %{receive: @short} = captured()
    end
  end

  describe "reranking" do
    defp reranker(opts) do
      %JinaRerankerM0{
        base_url: "https://rr.test",
        api_key: "test",
        req: capturing_req(%{"results" => [%{"index" => 0, "relevance_score" => 0.9}]})
      }
      |> struct!(opts)
    end

    test "given no timeout, then the provider default reaches Req" do
      assert {:ok, _} = JinaRerankerService.rerank(reranker([]), "q", ["a"])
      assert %{receive: @five_minutes} = captured()
    end

    test "given a call-site timeout, then it is accepted rather than refused as unknown" do
      assert {:ok, _} =
               JinaRerankerService.rerank(reranker([]), "q", ["a"], receive_timeout: @short)

      assert %{receive: @short} = captured()
    end
  end

  describe "uploads" do
    test "given no timeout, then five minutes is used instead of Req's short default" do
      req = capturing_req(%{"id" => "file-1", "filename" => "clip.mp4"})

      assert {:ok, _ref} = OpenAIUploadService.upload(req, "bytes", "video/mp4")
      assert %{receive: @five_minutes} = captured()
    end

    test "given a timeout, then the upload honors it" do
      req = capturing_req(%{"id" => "file-1", "filename" => "clip.mp4"})

      assert {:ok, _ref} =
               OpenAIUploadService.upload(req, "bytes", "video/mp4", receive_timeout: @short)

      assert %{receive: @short} = captured()
    end

    test "given a Gemini upload, then the timeout reaches the media endpoint" do
      req = capturing_req(%{"file" => %{"uri" => "files/1", "state" => "ACTIVE"}})

      assert {:ok, _ref} =
               GeminiUploadService.upload("AIza-test", "bytes", "video/mp4",
                 req: req,
                 upload_url: "https://upload.test/files",
                 receive_timeout: @short
               )

      assert %{receive: @short} = captured()
    end
  end
end
