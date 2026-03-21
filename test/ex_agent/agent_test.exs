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
      assert {:ok, %Message{role: :assistant, content: "Hello!"}} = Agent.chat(pid, "Hi")
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

      assert {:ok, %Message{content: "Found results for elixir"}} =
               Agent.chat(pid, "Search elixir")

      # Tool call flow produces 4 messages: user, assistant-tool-call, tool-result, assistant-final
      context = Agent.get_context(pid)
      assert length(context.messages) == 4
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

      assert {:ok, %Message{content: "I see the image"}} =
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
      assert {:error, {500, _}} = Agent.chat(pid, "Hi")
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
      assert {:ok, %Message{content: "Handled gracefully"}} = Agent.chat(pid, "Do something")
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
