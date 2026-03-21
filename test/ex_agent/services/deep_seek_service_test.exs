defmodule ExAgent.Services.DeepSeekServiceTest do
  use ExUnit.Case, async: true

  alias ExAgent.Services.DeepSeekService
  alias ExAgent.Message

  defp success_response(content) do
    %{
      "choices" => [
        %{"message" => %{"role" => "assistant", "content" => content}}
      ]
    }
  end

  defp build_req(plug_fn) do
    Req.new(plug: plug_fn)
  end

  # Happy path tests
  describe "chat/6 success" do
    test "returns assistant message for simple chat" do
      req = build_req(fn conn ->
        Req.Test.json(conn, success_response("Hello!"))
      end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")

      assert {:ok, %Message{role: :assistant, content: "Hello!"}} =
               DeepSeekService.chat(req, "deepseek-chat", [msg], [], nil)
    end

    test "sends system prompt when provided" do
      req = build_req(fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        parsed = Jason.decode!(body)
        [system | _] = parsed["messages"]
        assert system["role"] == "system"
        Req.Test.json(conn, success_response("Ok"))
      end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:ok, _} = DeepSeekService.chat(req, "deepseek-chat", [msg], [], "Be helpful")
    end

    test "uses correct model in request body" do
      req = build_req(fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        parsed = Jason.decode!(body)
        assert parsed["model"] == "deepseek-reasoner"
        Req.Test.json(conn, success_response("Ok"))
      end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:ok, _} = DeepSeekService.chat(req, "deepseek-reasoner", [msg], [], nil)
    end
  end

  # Bad path tests
  describe "chat/6 errors" do
    test "returns error for non-200 status" do
      req = build_req(fn conn ->
        conn |> Plug.Conn.send_resp(429, Jason.encode!(%{"error" => "rate limited"}))
      end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:error, {429, _}} = DeepSeekService.chat(req, "deepseek-chat", [msg], [], nil)
    end

    test "returns error for unexpected response" do
      req = build_req(fn conn ->
        Req.Test.json(conn, %{"no_choices" => true})
      end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:error, {:unexpected_response, _}} = DeepSeekService.chat(req, "deepseek-chat", [msg], [], nil)
    end

    test "returns error for server errors" do
      req = build_req(fn conn ->
        conn |> Plug.Conn.send_resp(503, Jason.encode!(%{"error" => "unavailable"}))
      end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:error, {503, _}} = DeepSeekService.chat(req, "deepseek-chat", [msg], [], nil)
    end
  end

  # Built-in tools tests
  describe "built_in_tools" do
    test "adds thinking mode to request body" do
      req = build_req(fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        parsed = Jason.decode!(body)
        assert parsed["thinking"] == %{"type" => "enabled"}
        Req.Test.json(conn, success_response("Thinking..."))
      end)

      {:ok, msg} = Message.new(role: :user, content: "Think about this")

      assert {:ok, _} =
               DeepSeekService.chat(req, "deepseek-reasoner", [msg], [], nil, built_in_tools: [:thinking])
    end

    test "ignores unknown built-in tools" do
      req = build_req(fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        parsed = Jason.decode!(body)
        refute Map.has_key?(parsed, "thinking")
        Req.Test.json(conn, success_response("Ok"))
      end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")

      assert {:ok, _} =
               DeepSeekService.chat(req, "deepseek-chat", [msg], [], nil, built_in_tools: [:unknown_tool])
    end
  end
end
