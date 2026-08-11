defmodule ExAgent.AgentTest do
  use ExUnit.Case, async: true

  alias ExAgent.{Agent, Context, Message, Skill, Tool}
  alias ExAgent.Providers.OpenAI

  defp success_response(content) do
    %{
      "choices" => [
        %{"message" => %{"role" => "assistant", "content" => content}}
      ]
    }
  end

  defp tool_call_response(name, args) do
    %{
      "choices" => [
        %{
          "message" => %{
            "role" => "assistant",
            "content" => nil,
            "tool_calls" => [
              %{
                "id" => "call_1",
                "type" => "function",
                "function" => %{"name" => name, "arguments" => Jason.encode!(args)}
              }
            ]
          }
        }
      ]
    }
  end

  defp build_provider(plug_fn) do
    %OpenAI{
      api_key: "test-key",
      model: "gpt-4o",
      base_url: "https://api.openai.com/v1",
      system_prompt: nil,
      tools: [],
      req: Req.new(plug: plug_fn)
    }
  end

  # Happy path tests
  describe "chat/2" do
    test "returns assistant response for simple message" do
      provider =
        build_provider(fn conn ->
          Req.Test.json(conn, success_response("Hello!"))
        end)

      {:ok, pid} = Agent.start_link(provider: provider)
      assert {:ok, %ExAgent.Response{content: "Hello!"}} = Agent.chat(pid, "Hi")
    end

    test "maintains conversation context across messages" do
      call_count = :counters.new(1, [:atomics])

      provider =
        build_provider(fn conn ->
          :counters.add(call_count, 1, 1)
          count = :counters.get(call_count, 1)

          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)

          if count == 2 do
            # Second call should have both user messages
            user_msgs = Enum.filter(parsed["messages"], &(&1["role"] == "user"))
            assert length(user_msgs) == 2
          end

          Req.Test.json(conn, success_response("Reply #{count}"))
        end)

      {:ok, pid} = Agent.start_link(provider: provider)
      {:ok, _} = Agent.chat(pid, "First")
      {:ok, msg} = Agent.chat(pid, "Second")
      assert msg.content == "Reply 2"
    end

    test "executes tools and returns final response" do
      call_count = :counters.new(1, [:atomics])

      provider =
        build_provider(fn conn ->
          :counters.add(call_count, 1, 1)
          count = :counters.get(call_count, 1)

          if count == 1 do
            Req.Test.json(conn, tool_call_response("search", %{"query" => "elixir"}))
          else
            Req.Test.json(conn, success_response("Found results for elixir"))
          end
        end)

      {:ok, tool} =
        Tool.new(
          name: "search",
          description: "Search",
          parameters: %{},
          function: fn %{"query" => q} -> {:ok, "Results for #{q}"} end
        )

      {:ok, pid} = Agent.start_link(provider: provider, tools: [tool])

      assert {:ok, %ExAgent.Response{content: "Found results for elixir"}} =
               Agent.chat(pid, "Search elixir")

      # Tool call flow produces 4 messages: user, assistant-tool-call, tool-result, assistant-final
      context = Agent.get_context(pid)
      assert length(context.messages) == 4
    end

    # `ExAgent.Tool`'s :function is typed `(map() -> any())` and its own doctest
    # returns a bare `:ok`, so a tool returning an unwrapped value must be taken
    # as the result rather than crashing the loop with a CaseClauseError.
    test "accepts a tool that returns a bare value instead of an {:ok, _} tuple" do
      call_count = :counters.new(1, [:atomics])

      provider =
        build_provider(fn conn ->
          :counters.add(call_count, 1, 1)

          if :counters.get(call_count, 1) == 1 do
            Req.Test.json(conn, tool_call_response("weather", %{"city" => "Lisbon"}))
          else
            {:ok, body, conn} = Plug.Conn.read_body(conn)
            # The bare value still reaches the model as the tool result.
            assert body =~ "Lisbon: 22C and sunny"
            Req.Test.json(conn, success_response("It is 22C in Lisbon"))
          end
        end)

      {:ok, tool} =
        Tool.new(
          name: "weather",
          description: "Weather",
          parameters: %{},
          function: fn %{"city" => city} -> "#{city}: 22C and sunny" end
        )

      {:ok, pid} = Agent.start_link(provider: provider, tools: [tool])

      assert {:ok, %ExAgent.Response{content: "It is 22C in Lisbon"}} =
               Agent.chat(pid, "Weather in Lisbon?")
    end

    test "given a tool returning a bare value, then streaming folds it in too" do
      # chat_stream/3 resolves tool turns non-streamed first and only streams the
      # final turn, so branch on what the request actually asks for.
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)

          cond do
            parsed["stream"] ->
              conn
              |> Plug.Conn.put_resp_content_type("text/event-stream")
              |> Plug.Conn.send_resp(
                200,
                ~s(data: {"choices":[{"delta":{"content":"22C in Porto"}}]}\n\ndata: [DONE]\n\n)
              )

            Enum.any?(parsed["messages"], &(&1["role"] == "tool")) ->
              assert body =~ "Porto: 22C and sunny"
              Req.Test.json(conn, success_response("done"))

            true ->
              Req.Test.json(conn, tool_call_response("weather", %{"city" => "Porto"}))
          end
        end)

      {:ok, tool} =
        Tool.new(
          name: "weather",
          description: "Weather",
          parameters: %{},
          function: fn %{"city" => city} -> "#{city}: 22C and sunny" end
        )

      {:ok, pid} = Agent.start_link(provider: provider, tools: [tool])

      assert {:ok, %ExAgent.Response{content: "22C in Porto"}} =
               pid |> Agent.chat_stream("Weather in Porto?") |> ExAgent.collect()
    end

    test "passes file attachments to message" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [msg] = parsed["messages"]
          # With attachments, content becomes multipart array
          assert is_list(msg["content"])
          Req.Test.json(conn, success_response("I see the image"))
        end)

      {:ok, pid} = Agent.start_link(provider: provider)

      assert {:ok, %ExAgent.Response{content: "I see the image"}} =
               Agent.chat(pid, "Describe this",
                 files: [%{data: "fake_png", mime_type: "image/png"}]
               )
    end

    test "passes built_in_tools to provider" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          assert parsed["web_search_options"] == %{}
          Req.Test.json(conn, success_response("Search results"))
        end)

      {:ok, pid} = Agent.start_link(provider: provider)

      assert {:ok, _} = Agent.chat(pid, "Search the web", built_in_tools: [:web_search])
    end

    test "uses agent-level built_in_tools as default" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          assert parsed["web_search_options"] == %{}
          Req.Test.json(conn, success_response("Search results"))
        end)

      {:ok, pid} = Agent.start_link(provider: provider, built_in_tools: [:web_search])

      assert {:ok, _} = Agent.chat(pid, "Search the web")
    end
  end

  # Bad path tests
  describe "chat/2 error handling" do
    test "returns error when LLM call fails" do
      provider =
        build_provider(fn conn ->
          conn |> Plug.Conn.send_resp(500, Jason.encode!(%{"error" => "internal"}))
        end)

      {:ok, pid} = Agent.start_link(provider: provider)

      assert {:error, %ExAgent.Error{type: :server, status: 500, retryable?: true}} =
               Agent.chat(pid, "Hi")
    end

    test "returns an error instead of crashing when an attachment is malformed" do
      provider = build_provider(fn conn -> Req.Test.json(conn, success_response("Hi")) end)
      {:ok, pid} = Agent.start_link(provider: provider)

      assert {:error, %ExAgent.Error{type: :invalid_request} = error} =
               Agent.chat(pid, "Look", files: [%{path: "/nonexistent/nope.png"}])

      assert error.message =~ "failed to read file"

      # The agent survived and is still usable.
      assert Process.alive?(pid)
      assert {:ok, _} = Agent.chat(pid, "Hi")
    end

    test "returns an error when an attachment mime type cannot be inferred" do
      provider = build_provider(fn conn -> Req.Test.json(conn, success_response("Hi")) end)
      {:ok, pid} = Agent.start_link(provider: provider)

      assert {:error, %ExAgent.Error{type: :invalid_request} = error} =
               Agent.chat(pid, "Look", files: [%{data: "not a known format"}])

      assert error.message =~ ":mime_type"
      assert Process.alive?(pid)
    end

    @tag :capture_log
    test "returns an %ExAgent.Error{} when the request task crashes" do
      provider = build_provider(fn _conn -> raise "boom" end)

      {:ok, pid} = Agent.start_link(provider: provider)

      assert {:error, %ExAgent.Error{type: :server} = error} = Agent.chat(pid, "Hi")
      assert error.message =~ "agent task failed"
      refute error.raw == nil

      # The agent survives the crash and is ready for the next turn.
      assert :sys.get_state(pid).status == :idle
    end

    test "handles unknown tool gracefully" do
      call_count = :counters.new(1, [:atomics])

      provider =
        build_provider(fn conn ->
          :counters.add(call_count, 1, 1)
          count = :counters.get(call_count, 1)

          if count == 1 do
            Req.Test.json(conn, tool_call_response("unknown_tool", %{}))
          else
            Req.Test.json(conn, success_response("Handled gracefully"))
          end
        end)

      {:ok, pid} = Agent.start_link(provider: provider)

      assert {:ok, %ExAgent.Response{content: "Handled gracefully"}} =
               Agent.chat(pid, "Do something")
    end

    test "returns error when max tool iterations reached" do
      provider =
        build_provider(fn conn ->
          Req.Test.json(conn, tool_call_response("loop", %{}))
        end)

      {:ok, tool} =
        Tool.new(
          name: "loop",
          description: "Loop forever",
          parameters: %{},
          function: fn _ -> {:ok, "again"} end
        )

      {:ok, pid} = Agent.start_link(provider: provider, tools: [tool])
      assert {:error, :max_tool_iterations_reached} = Agent.chat(pid, "Loop")
    end
  end

  # Edge case tests
  describe "get_context/1 and reset/1" do
    test "returns empty context initially" do
      provider = build_provider(fn conn -> Req.Test.json(conn, success_response("Ok")) end)
      {:ok, pid} = Agent.start_link(provider: provider)

      context = Agent.get_context(pid)
      assert context.messages == []
    end

    test "returns context with messages after chat" do
      provider = build_provider(fn conn -> Req.Test.json(conn, success_response("Hello!")) end)
      {:ok, pid} = Agent.start_link(provider: provider)

      {:ok, _} = Agent.chat(pid, "Hi")
      context = Agent.get_context(pid)
      assert length(context.messages) == 2
    end

    test "reset clears the context" do
      provider = build_provider(fn conn -> Req.Test.json(conn, success_response("Hello!")) end)
      {:ok, pid} = Agent.start_link(provider: provider)

      {:ok, _} = Agent.chat(pid, "Hi")
      Agent.reset(pid)
      # Give cast time to process
      :timer.sleep(10)
      context = Agent.get_context(pid)
      assert context.messages == []
    end
  end

  describe "load_skill/2" do
    test "dynamically adds a skill to the agent" do
      call_count = :counters.new(1, [:atomics])

      provider =
        build_provider(fn conn ->
          :counters.add(call_count, 1, 1)
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)

          # After skill loads, system prompt should be the skill's
          system_msgs = Enum.filter(parsed["messages"], &(&1["role"] == "system"))

          if length(system_msgs) > 0 do
            [sys | _] = system_msgs
            assert sys["content"] == "You are a SQL expert"
          end

          Req.Test.json(conn, success_response("SQL response"))
        end)

      provider_with_prompt = %{provider | system_prompt: nil}
      {:ok, pid} = Agent.start_link(provider: provider_with_prompt)

      {:ok, skill} =
        Skill.new(
          name: "sql",
          system_prompt: "You are a SQL expert",
          activation_fn: fn _ctx -> true end
        )

      Agent.load_skill(pid, skill)
      :timer.sleep(10)

      {:ok, msg} = Agent.chat(pid, "SELECT * FROM users")
      assert msg.content == "SQL response"
    end
  end

  # A one-shot gate: the plug blocks in wait/1 until release/1 is called.
  # Handles either ordering (wait-then-release or release-then-wait).
  defmodule Gate do
    def start_link, do: {:ok, spawn_link(fn -> loop(nil, false) end)}

    def wait(gate) do
      send(gate, {:wait, self()})

      receive do
        :go -> :ok
      after
        5_000 -> :ok
      end
    end

    def release(gate), do: send(gate, :release)

    defp loop(waiter, released) do
      receive do
        {:wait, pid} ->
          if released do
            send(pid, :go)
            loop(nil, false)
          else
            loop(pid, released)
          end

        :release ->
          if waiter do
            send(waiter, :go)
            loop(nil, false)
          else
            loop(waiter, true)
          end
      end
    end
  end

  defp wait_for_processing(pid) do
    if :sys.get_state(pid).status == :processing do
      :ok
    else
      Process.sleep(2)
      wait_for_processing(pid)
    end
  end

  describe "non-blocking dispatch" do
    defp gated_provider(gate) do
      build_provider(fn conn ->
        Gate.wait(gate)
        Req.Test.json(conn, success_response("done"))
      end)
    end

    test "stays responsive to get_context while a chat is in flight" do
      {:ok, gate} = Gate.start_link()
      {:ok, pid} = Agent.start_link(provider: gated_provider(gate))

      caller = Task.async(fn -> Agent.chat(pid, "Hi") end)
      wait_for_processing(pid)

      # Chat is blocked in the provider; the GenServer must still answer reads.
      context = Agent.get_context(pid)
      assert length(context.messages) == 1
      assert hd(context.messages).role == :user

      Gate.release(gate)
      assert {:ok, %ExAgent.Response{content: "done"}} = Task.await(caller)
    end

    test "returns :busy when a second chat starts while one is processing" do
      {:ok, gate} = Gate.start_link()
      {:ok, pid} = Agent.start_link(provider: gated_provider(gate))

      caller = Task.async(fn -> Agent.chat(pid, "First") end)
      wait_for_processing(pid)

      assert {:error, :busy} = Agent.chat(pid, "Second")

      Gate.release(gate)
      assert {:ok, _} = Task.await(caller)
    end

    test "recovers to idle after a chat completes and accepts the next one" do
      {:ok, gate} = Gate.start_link()
      {:ok, pid} = Agent.start_link(provider: gated_provider(gate))

      c1 = Task.async(fn -> Agent.chat(pid, "First") end)
      Gate.release(gate)
      assert {:ok, _} = Task.await(c1)

      c2 = Task.async(fn -> Agent.chat(pid, "Second") end)
      Gate.release(gate)
      assert {:ok, _} = Task.await(c2)
    end
  end

  describe "chat_stream/3" do
    defp sse(frames) do
      body = Enum.map_join(frames, "", fn f -> "data: #{Jason.encode!(f)}\n\n" end)
      body <> "data: [DONE]\n\n"
    end

    defp delta(content), do: %{"choices" => [%{"delta" => %{"content" => content}}]}

    defp stream_texts(stream) do
      stream
      |> Enum.filter(&(&1.type == :text_delta))
      |> Enum.map(& &1.text)
    end

    defp sse_plug(body) do
      fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_resp(200, body)
      end
    end

    test "streams text chunks when no tools are configured" do
      provider = build_provider(sse_plug(sse([delta("Hel"), delta("lo"), delta("!")])))
      {:ok, pid} = Agent.start_link(provider: provider)

      assert ["Hel", "lo", "!"] = pid |> Agent.chat_stream("Hi") |> stream_texts()
    end

    test "commits the streamed response to context once consumed" do
      provider = build_provider(sse_plug(sse([delta("Hi "), delta("there")])))
      {:ok, pid} = Agent.start_link(provider: provider)

      pid |> Agent.chat_stream("Hello") |> Stream.run()

      context = Agent.get_context(pid)
      # user message + streamed assistant message
      assert length(context.messages) == 2
      assert %Message{role: :assistant, content: "Hi there"} = List.last(context.messages)
    end

    test "returns to idle after streaming, accepting the next request" do
      provider = build_provider(sse_plug(sse([delta("ok")])))
      {:ok, pid} = Agent.start_link(provider: provider)

      pid |> Agent.chat_stream("first") |> Stream.run()
      assert :sys.get_state(pid).status == :idle
      assert ["ok"] = pid |> Agent.chat_stream("second") |> stream_texts()
    end

    test "resolves tool calls first, then streams the final turn" do
      call_count = :counters.new(1, [:atomics])

      provider =
        build_provider(fn conn ->
          :counters.add(call_count, 1, 1)
          count = :counters.get(call_count, 1)

          cond do
            # First (non-streamed) call: model asks for a tool.
            count == 1 ->
              Req.Test.json(conn, tool_call_response("search", %{"query" => "elixir"}))

            # Second (non-streamed) call detects the final turn (text).
            count == 2 ->
              Req.Test.json(conn, success_response("final"))

            # Third call is the streamed final turn.
            true ->
              conn
              |> Plug.Conn.put_resp_content_type("text/event-stream")
              |> Plug.Conn.send_resp(200, sse([delta("Found "), delta("elixir")]))
          end
        end)

      {:ok, tool} =
        Tool.new(
          name: "search",
          description: "Search",
          parameters: %{},
          function: fn %{"query" => q} -> {:ok, "Results for #{q}"} end
        )

      {:ok, pid} = Agent.start_link(provider: provider, tools: [tool])

      assert ["Found ", "elixir"] =
               pid |> Agent.chat_stream("Search elixir") |> stream_texts()

      # user, assistant-tool-call, tool-result, streamed assistant
      context = Agent.get_context(pid)
      assert length(context.messages) == 4
      assert %Message{role: :assistant, content: "Found elixir"} = List.last(context.messages)
    end

    test "yields an error chunk instead of raising when the agent is busy" do
      {:ok, gate} = Gate.start_link()

      provider =
        build_provider(fn conn ->
          Gate.wait(gate)
          Req.Test.json(conn, success_response("done"))
        end)

      {:ok, pid} = Agent.start_link(provider: provider)
      caller = Task.async(fn -> Agent.chat(pid, "First") end)
      wait_for_processing(pid)

      assert [%ExAgent.Chunk{type: :done, finish_reason: :error, error: error}] =
               pid |> Agent.chat_stream("Second") |> Enum.to_list()

      assert error.message =~ "processing another request"

      Gate.release(gate)
      assert {:ok, _} = Task.await(caller)
    end
  end

  describe "receive_handoff" do
    test "updates agent context via handoff" do
      provider = build_provider(fn conn -> Req.Test.json(conn, success_response("Ok")) end)
      {:ok, pid} = Agent.start_link(provider: provider)

      {:ok, msg} = Message.new(role: :user, content: "Transferred context")
      handoff_context = Context.new(messages: [msg])

      GenServer.cast(pid, {:receive_handoff, handoff_context})
      :timer.sleep(10)

      context = Agent.get_context(pid)
      assert length(context.messages) == 1
      assert hd(context.messages).content == "Transferred context"
    end
  end
end
