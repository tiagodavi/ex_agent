defmodule ExAgent.Patterns.RouterTest do
  use ExUnit.Case, async: true

  alias ExAgent.Patterns.Router
  alias ExAgent.Agent
  alias ExAgent.Providers.OpenAI

  defp success_response(content) do
    %{
      "choices" => [
        %{"message" => %{"role" => "assistant", "content" => content}}
      ]
    }
  end

  defp build_provider(plug_fn) do
    %OpenAI{
      api_key: "test",
      model: "gpt-4o",
      base_url: "https://api.openai.com/v1",
      system_prompt: nil,
      tools: [],
      req: Req.new(plug: plug_fn)
    }
  end

  defp start_agent(response_content) do
    provider =
      build_provider(fn conn ->
        Req.Test.json(conn, success_response(response_content))
      end)

    {:ok, pid} = Agent.start_link(provider: provider)
    pid
  end

  # Happy path tests
  describe "classify/2" do
    test "returns matching routes" do
      routes = [
        %{name: "code", agent: self(), match_fn: &String.contains?(&1, "code")},
        %{name: "write", agent: self(), match_fn: &String.contains?(&1, "write")}
      ]

      matched = Router.classify("write a poem", routes)
      assert length(matched) == 1
      assert hd(matched).name == "write"
    end

    test "returns multiple matching routes" do
      routes = [
        %{name: "code", agent: self(), match_fn: &String.contains?(&1, "code")},
        %{name: "review", agent: self(), match_fn: &String.contains?(&1, "code")}
      ]

      matched = Router.classify("review this code", routes)
      assert length(matched) == 2
    end

    test "returns all routes when all match" do
      routes = [
        %{name: "a", agent: self(), match_fn: fn _ -> true end},
        %{name: "b", agent: self(), match_fn: fn _ -> true end}
      ]

      assert length(Router.classify("anything", routes)) == 2
    end
  end

  # Bad path tests
  describe "classify/2 no matches" do
    test "returns empty list when nothing matches" do
      routes = [
        %{name: "code", agent: self(), match_fn: &String.contains?(&1, "code")}
      ]

      assert Router.classify("hello world", routes) == []
    end

    test "returns empty list for empty routes" do
      assert Router.classify("anything", []) == []
    end

    test "route/2 returns error when no routes match" do
      routes = [
        %{name: "code", agent: self(), match_fn: &String.contains?(&1, "code")}
      ]

      assert {:error, :no_matching_routes} = Router.route("hello", routes: routes)
    end
  end

  # Edge case / integration tests
  describe "route/2 integration" do
    test "routes to single agent and returns synthesized result" do
      agent_pid = start_agent("Code review result")

      routes = [
        %{name: "coder", agent: agent_pid, match_fn: &String.contains?(&1, "code")}
      ]

      assert {:ok, result} = Router.route("review this code", routes: routes)
      assert String.contains?(result, "Code review result")
      assert String.contains?(result, "## coder")
    end

    test "routes to multiple agents in parallel and synthesizes" do
      coder_pid = start_agent("Code analysis")
      writer_pid = start_agent("Writing analysis")

      routes = [
        %{name: "coder", agent: coder_pid, match_fn: fn _ -> true end},
        %{name: "writer", agent: writer_pid, match_fn: fn _ -> true end}
      ]

      assert {:ok, result} = Router.route("analyze this", routes: routes)
      assert String.contains?(result, "Code analysis")
      assert String.contains?(result, "Writing analysis")
    end

    test "uses custom synthesizer" do
      agent_pid = start_agent("Agent response")

      synthesizer = fn _input, results ->
        results |> Enum.map(fn {name, content} -> "#{name}: #{content}" end) |> Enum.join("; ")
      end

      routes = [
        %{name: "test", agent: agent_pid, match_fn: fn _ -> true end}
      ]

      assert {:ok, "test: Agent response"} =
               Router.route("query", routes: routes, synthesizer: synthesizer)
    end
  end

  describe "dispatch/3" do
    test "returns results from all agents" do
      pid1 = start_agent("Result 1")
      pid2 = start_agent("Result 2")

      routes = [
        %{name: "agent1", agent: pid1, match_fn: fn _ -> true end},
        %{name: "agent2", agent: pid2, match_fn: fn _ -> true end}
      ]

      results = Router.dispatch("query", routes)
      assert length(results) == 2
      assert {"agent1", "Result 1"} in results
      assert {"agent2", "Result 2"} in results
    end
  end
end
