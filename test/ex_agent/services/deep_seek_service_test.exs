defmodule ExAgent.Services.DeepSeekServiceTest do
  use ExUnit.Case, async: true

  alias ExAgent.Services.DeepSeekService
  alias ExAgent.Providers.DeepSeek
  alias ExAgent.Message

  defp success_response(content) do
    %{
      "choices" => [
        %{"message" => %{"role" => "assistant", "content" => content}}
      ]
    }
  end

  defp build_provider(plug_fn, opts \\ []) do
    req = Req.new(plug: plug_fn)

    %DeepSeek{
      api_key: "sk-test",
      model: opts[:model] || "deepseek-chat",
      base_url: "https://api.deepseek.com/v1",
      system_prompt: opts[:system_prompt],
      tools: opts[:tools] || [],
      req: req
    }
  end

  # Happy path tests
  describe "chat/3 success" do
    test "returns assistant message for simple chat" do
      provider =
        build_provider(fn conn ->
          Req.Test.json(conn, success_response("Hello!"))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")

      assert {:ok, %Message{role: :assistant, content: "Hello!"}} =
               DeepSeekService.chat(provider, [msg])
    end

    test "sends system prompt when provided" do
      provider =
        build_provider(
          fn conn ->
            {:ok, body, conn} = Plug.Conn.read_body(conn)
            parsed = Jason.decode!(body)
            [system | _] = parsed["messages"]
            assert system["role"] == "system"
            Req.Test.json(conn, success_response("Ok"))
          end,
          system_prompt: "Be helpful"
        )

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:ok, _} = DeepSeekService.chat(provider, [msg])
    end

    test "uses correct model in request body" do
      provider =
        build_provider(
          fn conn ->
            {:ok, body, conn} = Plug.Conn.read_body(conn)
            parsed = Jason.decode!(body)
            assert parsed["model"] == "deepseek-reasoner"
            Req.Test.json(conn, success_response("Ok"))
          end,
          model: "deepseek-reasoner"
        )

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:ok, _} = DeepSeekService.chat(provider, [msg])
    end
  end

  # Bad path tests
  describe "chat/3 errors" do
    test "returns error for non-200 status" do
      provider =
        build_provider(fn conn ->
          conn |> Plug.Conn.send_resp(429, Jason.encode!(%{"error" => "rate limited"}))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:error, {429, _}} = DeepSeekService.chat(provider, [msg])
    end

    test "returns error for unexpected response" do
      provider =
        build_provider(fn conn ->
          Req.Test.json(conn, %{"no_choices" => true})
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")

      assert {:error, {:unexpected_response, _}} =
               DeepSeekService.chat(provider, [msg])
    end

    test "returns error for server errors" do
      provider =
        build_provider(fn conn ->
          conn |> Plug.Conn.send_resp(503, Jason.encode!(%{"error" => "unavailable"}))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:error, {503, _}} = DeepSeekService.chat(provider, [msg])
    end
  end

  describe "chat/3 attachments" do
    test "raises ArgumentError when attachments are provided" do
      provider =
        build_provider(fn conn ->
          Req.Test.json(conn, success_response("Ok"))
        end)

      {:ok, msg} =
        Message.new(
          role: :user,
          content: "Describe this",
          attachments: [%{data: "fake_png", mime_type: "image/png"}]
        )

      assert_raise ArgumentError, ~r/DeepSeek does not support file attachments/, fn ->
        DeepSeekService.chat(provider, [msg])
      end
    end
  end

  # Built-in tools tests
  describe "built_in_tools" do
    test "adds thinking mode to request body" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          assert parsed["thinking"] == %{"type" => "enabled"}
          Req.Test.json(conn, success_response("Thinking..."))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Think about this")

      assert {:ok, _} =
               DeepSeekService.chat(provider, [msg], built_in_tools: [:thinking])
    end

    test "ignores unknown built-in tools" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          refute Map.has_key?(parsed, "thinking")
          Req.Test.json(conn, success_response("Ok"))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")

      assert {:ok, _} =
               DeepSeekService.chat(provider, [msg], built_in_tools: [:unknown_tool])
    end
  end
end
