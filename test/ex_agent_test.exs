defmodule ExAgentTest do
  use ExUnit.Case

  alias ExAgent.Message
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

  describe "public API" do
    test "start_agent/1 and chat/2 work through the facade" do
      provider = build_provider(fn conn ->
        Req.Test.json(conn, success_response("Hello from facade!"))
      end)

      {:ok, pid} = ExAgent.start_agent(provider: provider)
      assert {:ok, %Message{content: "Hello from facade!"}} = ExAgent.chat(pid, "Hi")
    end

    test "stop_agent/1 terminates the agent" do
      provider = build_provider(fn conn ->
        Req.Test.json(conn, success_response("Ok"))
      end)

      {:ok, pid} = ExAgent.start_agent(provider: provider)
      assert :ok = ExAgent.stop_agent(pid)
      refute Process.alive?(pid)
    end

    test "get_context/1 returns the conversation context" do
      provider = build_provider(fn conn ->
        Req.Test.json(conn, success_response("Response"))
      end)

      {:ok, pid} = ExAgent.start_agent(provider: provider)
      {:ok, _} = ExAgent.chat(pid, "Hello")

      context = ExAgent.get_context(pid)
      assert length(context.messages) == 2
    end
  end
end
