defmodule ExAgent.Patterns.HandoffTest do
  use ExUnit.Case, async: true

  alias ExAgent.Patterns.Handoff
  alias ExAgent.{Agent, Context, Message}
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

  # Happy path tests
  describe "build_handoff_tool/3" do
    test "creates a tool with correct name and description" do
      tool = Handoff.build_handoff_tool("support", self(), "Transfer to support agent")
      assert tool.name == "handoff_to_support"
      assert tool.description == "Transfer to support agent"
    end

    test "tool function returns handoff tuple" do
      tool = Handoff.build_handoff_tool("support", self(), "Transfer")
      result = tool.function.(%{"summary" => "User needs billing help"})
      assert {:handoff, _target, %Context{}} = result
    end

    test "handoff context includes summary message" do
      tool = Handoff.build_handoff_tool("support", self(), "Transfer")
      {:handoff, _target, context} = tool.function.(%{"summary" => "Billing issue"})

      [msg] = context.messages
      assert msg.role == :user
      assert String.contains?(msg.content, "Billing issue")
    end
  end

  # Bad path / transfer_context tests
  describe "transfer_context/2" do
    test "adds handoff summary as user message" do
      context = Context.new()
      new_context = Handoff.transfer_context(context, %{"summary" => "User wants refund"})

      assert length(new_context.messages) == 1
      [msg] = new_context.messages
      assert msg.role == :user
      assert String.contains?(msg.content, "User wants refund")
    end

    test "preserves existing messages in context" do
      {:ok, existing_msg} = Message.new(role: :user, content: "Hello")
      context = Context.new(messages: [existing_msg])

      new_context = Handoff.transfer_context(context, %{"summary" => "Summary"})
      assert length(new_context.messages) == 2
    end

    test "includes handoff prefix in message" do
      context = Handoff.transfer_context(Context.new(), %{"summary" => "Test"})
      [msg] = context.messages
      assert String.starts_with?(msg.content, "Handoff received.")
    end
  end

  # Edge case tests — integration with Agent
  describe "execute_handoff/2" do
    test "sends context to target agent via cast" do
      provider = build_provider(fn conn -> Req.Test.json(conn, success_response("Ok")) end)
      {:ok, target_pid} = Agent.start_link(provider: provider)

      {:ok, msg} = Message.new(role: :system, content: "Handoff context")
      context = Context.new(messages: [msg])

      Handoff.execute_handoff(target_pid, context)
      :timer.sleep(10)

      received_context = Agent.get_context(target_pid)
      assert length(received_context.messages) == 1
      assert hd(received_context.messages).content == "Handoff context"
    end

    test "agent can respond after receiving handoff" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          msgs = parsed["messages"]
          # Should have the handoff system message + new user message
          assert length(msgs) >= 2
          Req.Test.json(conn, success_response("Handled after handoff"))
        end)

      {:ok, target_pid} = Agent.start_link(provider: provider)

      {:ok, sys_msg} = Message.new(role: :system, content: "Handoff: billing issue")
      context = Context.new(messages: [sys_msg])
      Handoff.execute_handoff(target_pid, context)
      :timer.sleep(10)

      {:ok, response} = Agent.chat(target_pid, "What about my refund?")
      assert response.content == "Handled after handoff"
    end

    test "handoff tool triggers handoff response from agent" do
      call_count = :counters.new(1, [:atomics])

      provider =
        build_provider(fn conn ->
          :counters.add(call_count, 1, 1)
          count = :counters.get(call_count, 1)

          if count == 1 do
            response = %{
              "choices" => [
                %{
                  "message" => %{
                    "tool_calls" => [
                      %{
                        "function" => %{
                          "name" => "handoff_to_support",
                          "arguments" => Jason.encode!(%{"summary" => "Needs help"})
                        }
                      }
                    ]
                  }
                }
              ]
            }

            Req.Test.json(conn, response)
          else
            Req.Test.json(conn, success_response("Should not reach here"))
          end
        end)

      target_provider =
        build_provider(fn conn -> Req.Test.json(conn, success_response("Support here")) end)

      {:ok, target_pid} = Agent.start_link(provider: target_provider)

      handoff_tool = Handoff.build_handoff_tool("support", target_pid, "Transfer to support")
      {:ok, agent_pid} = Agent.start_link(provider: provider, tools: [handoff_tool])

      result = Agent.chat(agent_pid, "I need support")
      assert {:handoff, ^target_pid, %Context{}} = result
    end
  end
end
