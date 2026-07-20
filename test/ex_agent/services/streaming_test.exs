defmodule ExAgent.Services.StreamingTest do
  use ExUnit.Case, async: true

  alias ExAgent.Services.Streaming
  alias ExAgent.Providers.{DeepSeek, Gemini, OpenAI}
  alias ExAgent.Message

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

  defp deepseek_provider(plug_fn, opts) do
    %DeepSeek{
      api_key: "sk-test",
      model: opts[:model] || "deepseek-chat",
      base_url: "http://x",
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

  defp user_msg(content) do
    {:ok, msg} = Message.new(role: :user, content: content)
    msg
  end

  # --- Happy path ---

  describe "stream/3 success" do
    test "OpenAI assembles multi-frame stream into text chunks" do
      body = sse([openai_frame("Hello"), openai_frame(" there"), openai_frame("!")])
      provider = openai_provider(sse_plug(body))

      assert ["Hello", " there", "!"] =
               provider |> OpenAI.stream([user_msg("hi")]) |> Enum.to_list()
    end

    test "DeepSeek reasoner streams text chunks" do
      body = sse([openai_frame("Let me think"), openai_frame(" ... done")])
      provider = deepseek_provider(sse_plug(body), model: "deepseek-reasoner")

      assert ["Let me think", " ... done"] =
               provider |> DeepSeek.stream([user_msg("2+2?")]) |> Enum.to_list()
    end

    test "Gemini parses alt=sse frames" do
      body = sse([gemini_frame("Elixir"), gemini_frame(" rocks")])
      provider = gemini_provider(sse_plug(body))

      assert ["Elixir", " rocks"] =
               provider |> Gemini.stream([user_msg("hi")]) |> Enum.to_list()
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
    test "raises StreamError on non-200 status when consumed" do
      provider = openai_provider(sse_plug(~s({"error":"unauthorized"}), 401))
      stream = OpenAI.stream(provider, [user_msg("hi")])

      assert_raise ExAgent.StreamError, fn -> Enum.to_list(stream) end
    end

    test "StreamError carries status and body" do
      provider = openai_provider(sse_plug(~s({"error":"nope"}), 403))

      error =
        assert_raise ExAgent.StreamError, fn ->
          OpenAI.stream(provider, [user_msg("hi")]) |> Enum.to_list()
        end

      assert error.status == 403
    end

    test "skips malformed JSON frames instead of crashing" do
      body = "data: {not valid json}\n\n" <> sse([openai_frame("recovered")])
      provider = openai_provider(sse_plug(body))

      assert ["recovered"] = provider |> OpenAI.stream([user_msg("hi")]) |> Enum.to_list()
    end
  end

  # --- Edge cases ---

  describe "stream/3 edge cases" do
    test "[DONE] terminates cleanly with no trailing content" do
      provider = openai_provider(sse_plug(sse([openai_frame("only")])))
      assert ["only"] = provider |> OpenAI.stream([user_msg("hi")]) |> Enum.to_list()
    end

    test "filters empty and content-less delta frames (e.g. tool-call frames)" do
      role_only = %{"choices" => [%{"delta" => %{"role" => "assistant"}}]}
      empty = openai_frame("")
      body = sse([role_only, empty, openai_frame("real")])
      provider = openai_provider(sse_plug(body))

      assert ["real"] = provider |> OpenAI.stream([user_msg("hi")]) |> Enum.to_list()
    end

    test "empty stream yields no chunks" do
      provider = openai_provider(sse_plug("data: [DONE]\n\n"))
      assert [] = provider |> OpenAI.stream([user_msg("hi")]) |> Enum.to_list()
    end
  end

  # --- Pure SSE framing helpers (split-across-chunks reassembly) ---

  describe "take_events/1" do
    test "holds back an incomplete trailing event" do
      assert {["data: a"], "data: b"} = Streaming.take_events("data: a\n\ndata: b")
    end

    test "reassembles an event split across two buffers" do
      # First buffer ends mid-event; the remainder carries into the next.
      {events1, rest1} = Streaming.take_events("data: {\"x\":1}\n\ndata: {\"y\"")
      assert events1 == ["data: {\"x\":1}"]

      {events2, rest2} = Streaming.take_events(rest1 <> ":2}\n\n")
      assert events2 == ["data: {\"y\":2}"]
      assert rest2 == ""
    end

    test "returns no complete events when buffer has none" do
      assert {[], "partial"} = Streaming.take_events("partial")
    end
  end

  describe "event_data/1" do
    test "extracts the data payload" do
      assert Streaming.event_data("data: {\"a\":1}") == "{\"a\":1}"
    end

    test "returns :done for the [DONE] sentinel" do
      assert Streaming.event_data("data: [DONE]") == :done
    end

    test "returns nil for events without a data line" do
      assert Streaming.event_data("event: ping") == nil
    end
  end

  # --- Dispatch through the Provider behaviour ---

  describe "ExAgent.Provider.stream/3" do
    test "routes to the provider module" do
      provider = openai_provider(sse_plug(sse([openai_frame("routed")])))
      assert ["routed"] = ExAgent.Provider.stream(provider, [user_msg("hi")]) |> Enum.to_list()
    end

    test "raises for a provider that does not implement stream/3" do
      assert_raise ArgumentError, ~r/does not support streaming/, fn ->
        ExAgent.Provider.stream(%NoStreamProvider{api_key: "x"}, [])
      end
    end
  end
end
