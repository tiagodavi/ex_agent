defmodule ExAgent.Patterns.SubagentsTest do
  use ExUnit.Case, async: true

  alias ExAgent.Patterns.Subagents
  alias ExAgent.Providers.OpenAI

  defp success_response(content) do
    %{
      "choices" => [
        %{"message" => %{"role" => "assistant", "content" => content}}
      ]
    }
  end

  defp build_spec(name, plug_fn) do
    %{
      name: name,
      description: "#{name} subagent",
      provider: %OpenAI{
        api_key: "test",
        model: "gpt-4o",
        base_url: "https://api.openai.com/v1",
        system_prompt: nil,
        tools: [],
        req: Req.new(plug: plug_fn)
      },
      system_prompt: "You are a #{name} expert",
      tools: []
    }
  end

  # Happy path tests
  describe "invoke_subagent/2" do
    test "returns successful response from subagent" do
      spec =
        build_spec("researcher", fn conn ->
          Req.Test.json(conn, success_response("Research findings"))
        end)

      assert {:ok, "Research findings"} =
               Subagents.invoke_subagent(spec, "Find info about Elixir")
    end

    test "sets system_prompt on the provider" do
      spec =
        build_spec("coder", fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [system | _] = parsed["messages"]
          assert system["content"] == "You are a coder expert"
          Req.Test.json(conn, success_response("Code result"))
        end)

      assert {:ok, "Code result"} = Subagents.invoke_subagent(spec, "Write a function")
    end

    test "context is isolated — fresh for each call" do
      spec =
        build_spec("isolated", fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          # Should only have system + 1 user message (no previous context)
          user_msgs = Enum.filter(parsed["messages"], &(&1["role"] == "user"))
          assert length(user_msgs) == 1
          Req.Test.json(conn, success_response("Fresh context"))
        end)

      {:ok, _} = Subagents.invoke_subagent(spec, "First call")
      {:ok, _} = Subagents.invoke_subagent(spec, "Second call")
    end
  end

  # Bad path tests
  describe "invoke_subagent/2 errors" do
    test "returns error when LLM call fails" do
      spec =
        build_spec("failing", fn conn ->
          conn |> Plug.Conn.send_resp(500, Jason.encode!(%{"error" => "internal"}))
        end)

      assert {:error, {500, _}} = Subagents.invoke_subagent(spec, "Do something")
    end

    test "handles tool_call response gracefully" do
      spec =
        build_spec("tool_caller", fn conn ->
          response = %{
            "choices" => [
              %{
                "message" => %{
                  "tool_calls" => [
                    %{"function" => %{"name" => "test", "arguments" => "{}"}}
                  ]
                }
              }
            ]
          }

          Req.Test.json(conn, response)
        end)

      assert {:ok, msg} = Subagents.invoke_subagent(spec, "Use a tool")
      assert String.contains?(msg, "tool call")
    end

    test "returns error for network failures" do
      spec =
        build_spec("network_fail", fn conn ->
          conn |> Plug.Conn.send_resp(503, "Service Unavailable")
        end)

      assert {:error, _} = Subagents.invoke_subagent(spec, "Hello")
    end
  end

  # Edge case tests
  describe "build_orchestrator_tools/1" do
    test "converts specs into Tool structs" do
      spec =
        build_spec("researcher", fn conn ->
          Req.Test.json(conn, success_response("Result"))
        end)

      tools = Subagents.build_orchestrator_tools([spec])
      assert length(tools) == 1
      [tool] = tools
      assert tool.name == "researcher"
      assert tool.description == "researcher subagent"
      assert is_function(tool.function, 1)
    end

    test "tool function invokes subagent when called" do
      spec =
        build_spec("worker", fn conn ->
          Req.Test.json(conn, success_response("Worker result"))
        end)

      [tool] = Subagents.build_orchestrator_tools([spec])
      assert {:ok, "Worker result"} = tool.function.(%{"query" => "Do work"})
    end

    test "builds multiple tools from multiple specs" do
      spec1 = build_spec("a", fn conn -> Req.Test.json(conn, success_response("A")) end)
      spec2 = build_spec("b", fn conn -> Req.Test.json(conn, success_response("B")) end)

      tools = Subagents.build_orchestrator_tools([spec1, spec2])
      assert length(tools) == 2
      assert Enum.map(tools, & &1.name) == ["a", "b"]
    end
  end

  describe "invoke_subagents_parallel/2" do
    test "invokes multiple subagents in parallel and returns results" do
      spec1 =
        build_spec("fast", fn conn -> Req.Test.json(conn, success_response("Fast result")) end)

      spec2 =
        build_spec("slow", fn conn -> Req.Test.json(conn, success_response("Slow result")) end)

      results = Subagents.invoke_subagents_parallel([{spec1, "query1"}, {spec2, "query2"}])

      assert length(results) == 2
      assert {"fast", {:ok, "Fast result"}} in results
      assert {"slow", {:ok, "Slow result"}} in results
    end
  end
end
