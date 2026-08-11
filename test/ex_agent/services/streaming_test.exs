defmodule ExAgent.Services.StreamingTest do
  use ExUnit.Case, async: true

  alias ExAgent.{Chunk, Message}
  alias ExAgent.Providers.{Gemini, OpenAI, OpenAICompatible}

  defmodule NoStreamProvider do
    defstruct [:api_key]
  end

  # --- SSE body builders (mirror each provider's streaming wire format) ---

  defp openai_frame(content), do: %{"choices" => [%{"delta" => %{"content" => content}}]}

  defp gemini_frame(text),
    do: %{"candidates" => [%{"content" => %{"parts" => [%{"text" => text}]}}]}

  defp sse(frames) do
    body = Enum.map_join(frames, "", fn f -> "data: #{Jason.encode!(f)}\n\n" end)
    body <> "data: [DONE]\n\n"
  end

  defp sse_plug(body, status \\ 200) do
    fn conn ->
      conn
      |> Plug.Conn.put_resp_content_type("text/event-stream")
      |> Plug.Conn.send_resp(status, body)
    end
  end

  defp openai_provider(plug_fn) do
    %OpenAI{
      api_key: "sk-test",
      model: "gpt-4o",
      base_url: "http://x",
      req: Req.new(plug: plug_fn)
    }
  end

  defp compatible_provider(plug_fn) do
    %OpenAICompatible{
      model: "Qwen/Qwen3-8B",
      base_url: "http://x/v1",
      req: Req.new(plug: plug_fn)
    }
  end

  defp gemini_provider(plug_fn) do
    %Gemini{
      api_key: "k",
      model: "gemini-2.0-flash",
      base_url: "http://x",
      req: Req.new(plug: plug_fn)
    }
  end

  # Streams now yield %ExAgent.Chunk{}; most of these assertions are about the
  # text they carry.
  defp texts(stream) do
    stream
    |> Enum.filter(&(&1.type == :text_delta))
    |> Enum.map(& &1.text)
  end

  defp user_msg(content) do
    {:ok, msg} = Message.new(role: :user, content: content)
    msg
  end

  # --- Happy path ---

  describe "stream/3 success" do
    test "OpenAI assembles multi-frame stream into text chunks" do
      body = sse([openai_frame("Hello"), openai_frame(" there"), openai_frame("!")])
      provider = openai_provider(sse_plug(body))

      assert ["Hello", " there", "!"] = provider |> OpenAI.stream([user_msg("hi")]) |> texts()
    end

    test "an OpenAI-compatible endpoint streams text chunks" do
      body = sse([openai_frame("Let me think"), openai_frame(" ... done")])
      provider = compatible_provider(sse_plug(body))

      assert ["Let me think", " ... done"] =
               provider |> OpenAICompatible.stream([user_msg("2+2?")]) |> texts()
    end

    test "Gemini parses alt=sse frames" do
      body = sse([gemini_frame("Elixir"), gemini_frame(" rocks")])
      provider = gemini_provider(sse_plug(body))

      assert ["Elixir", " rocks"] = provider |> Gemini.stream([user_msg("hi")]) |> texts()
    end

    test "sets stream:true and hits the streaming endpoint (Gemini)" do
      # Capture in a separate process: the streaming receive loop drains the
      # calling process's mailbox, so we can't send the path to self().
      {:ok, collector} = Agent.start_link(fn -> nil end)

      plug = fn conn ->
        Agent.update(collector, fn _ -> conn.request_path <> "?" <> (conn.query_string || "") end)

        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_resp(200, sse([gemini_frame("ok")]))
      end

      gemini_provider(plug) |> Gemini.stream([user_msg("hi")]) |> Enum.to_list()

      path = Agent.get(collector, & &1)
      assert path =~ ":streamGenerateContent"
      assert path =~ "alt=sse"
    end

    test "OpenAI sets stream:true in the request body" do
      {:ok, collector} = Agent.start_link(fn -> nil end)

      plug = fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        Agent.update(collector, fn _ -> Jason.decode!(body) end)

        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_resp(200, sse([openai_frame("ok")]))
      end

      openai_provider(plug) |> OpenAI.stream([user_msg("hi")]) |> Enum.to_list()
      assert Agent.get(collector, & &1)["stream"] == true
    end
  end

  # --- Failure path ---

  describe "stream/3 failures" do
    test "a non-200 status yields a terminal error chunk rather than raising" do
      provider = openai_provider(sse_plug(~s({"error":"unauthorized"}), 401))

      assert [%Chunk{type: :done, finish_reason: :error, error: error}] =
               provider |> OpenAI.stream([user_msg("hi")]) |> Enum.to_list()

      assert error.type == :auth
      assert error.status == 401
    end

    test "the error chunk carries a retryable flag for transient failures" do
      provider = openai_provider(sse_plug(~s({"error":"slow down"}), 429))

      assert [%Chunk{type: :done, error: error}] =
               provider |> OpenAI.stream([user_msg("hi")]) |> Enum.to_list()

      assert error.type == :rate_limit
      assert error.retryable?
    end

    test "skips malformed JSON frames instead of crashing" do
      body = "data: {not valid json}\n\n" <> sse([openai_frame("recovered")])
      provider = openai_provider(sse_plug(body))

      assert ["recovered"] = provider |> OpenAI.stream([user_msg("hi")]) |> texts()
    end
  end

  # --- Edge cases ---

  describe "stream/3 edge cases" do
    test "[DONE] terminates cleanly with no trailing content" do
      provider = openai_provider(sse_plug(sse([openai_frame("only")])))
      assert ["only"] = provider |> OpenAI.stream([user_msg("hi")]) |> texts()
    end

    test "filters empty and content-less delta frames (e.g. tool-call frames)" do
      role_only = %{"choices" => [%{"delta" => %{"role" => "assistant"}}]}
      empty = openai_frame("")
      body = sse([role_only, empty, openai_frame("real")])
      provider = openai_provider(sse_plug(body))

      assert ["real"] = provider |> OpenAI.stream([user_msg("hi")]) |> texts()
    end

    test "empty stream yields no chunks" do
      provider = openai_provider(sse_plug("data: [DONE]\n\n"))
      assert [] = provider |> OpenAI.stream([user_msg("hi")]) |> texts()
    end
  end

  # SSE framing helpers now live in ExAgent.SSE (see test/ex_agent/sse_test.exs).

  # --- Dispatch through the Provider behaviour ---

  describe "ExAgent.Provider.stream/3" do
    test "routes to the provider module" do
      provider = openai_provider(sse_plug(sse([openai_frame("routed")])))
      assert ["routed"] = ExAgent.Provider.stream(provider, [user_msg("hi")]) |> texts()
    end

    test "raises an unsupported ExAgent.Error for a provider without stream/3" do
      error =
        assert_raise ExAgent.Error, ~r/does not support streaming/, fn ->
          ExAgent.Provider.stream(%NoStreamProvider{api_key: "x"}, [])
        end

      assert error.type == :unsupported
      assert error.provider == NoStreamProvider
    end
  end
end
