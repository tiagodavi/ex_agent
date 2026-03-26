defmodule ExAgent.Services.OpenAIServiceTest do
  use ExUnit.Case, async: true

  alias ExAgent.Services.OpenAIService
  alias ExAgent.Providers.OpenAI
  alias ExAgent.{Message, Tool}

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
                "id" => "call_123",
                "type" => "function",
                "function" => %{"name" => name, "arguments" => Jason.encode!(args)}
              }
            ]
          }
        }
      ]
    }
  end

  defp build_provider(plug_fn, opts \\ []) do
    req = Req.new(plug: plug_fn)

    %OpenAI{
      api_key: "sk-test",
      model: opts[:model] || "gpt-4o",
      base_url: "https://api.openai.com/v1",
      temperature: opts[:temperature] || 0.6,
      max_tokens: opts[:max_tokens] || 512,
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
          Req.Test.json(conn, success_response("Hello there!"))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")

      assert {:ok, %Message{role: :assistant, content: "Hello there!"}} =
               OpenAIService.chat(provider, [msg])
    end

    test "sends system prompt when provided" do
      provider =
        build_provider(
          fn conn ->
            {:ok, body, conn} = Plug.Conn.read_body(conn)
            parsed = Jason.decode!(body)
            [system | _] = parsed["messages"]
            assert system["role"] == "system"
            assert system["content"] == "Be helpful"
            Req.Test.json(conn, success_response("Sure!"))
          end,
          system_prompt: "Be helpful"
        )

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:ok, _} = OpenAIService.chat(provider, [msg])
    end

    test "returns tool_call when LLM wants to invoke a tool" do
      provider =
        build_provider(fn conn ->
          Req.Test.json(conn, tool_call_response("search", %{"query" => "elixir"}))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Search for elixir")

      assert {:tool_call, "search", %{"query" => "elixir"}} =
               OpenAIService.chat(provider, [msg])
    end
  end

  # Bad path tests
  describe "chat/3 errors" do
    test "returns error for non-200 status" do
      provider =
        build_provider(fn conn ->
          conn
          |> Plug.Conn.send_resp(401, Jason.encode!(%{"error" => "unauthorized"}))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:error, {401, _}} = OpenAIService.chat(provider, [msg])
    end

    test "returns error for unexpected response format" do
      provider =
        build_provider(fn conn ->
          Req.Test.json(conn, %{"unexpected" => "format"})
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")

      assert {:error, {:unexpected_response, _}} =
               OpenAIService.chat(provider, [msg])
    end

    test "returns error for server errors" do
      provider =
        build_provider(fn conn ->
          conn
          |> Plug.Conn.send_resp(500, Jason.encode!(%{"error" => "internal"}))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:error, {500, _}} = OpenAIService.chat(provider, [msg])
    end
  end

  # Edge case tests
  describe "chat/3 edge cases" do
    test "sends tools in correct OpenAI format" do
      {:ok, tool} =
        Tool.new(
          name: "search",
          description: "Search the web",
          parameters: %{"type" => "object", "properties" => %{"q" => %{"type" => "string"}}},
          function: & &1
        )

      provider =
        build_provider(
          fn conn ->
            {:ok, body, conn} = Plug.Conn.read_body(conn)
            parsed = Jason.decode!(body)
            [sent_tool] = parsed["tools"]
            assert sent_tool["type"] == "function"
            assert sent_tool["function"]["name"] == "search"
            Req.Test.json(conn, success_response("Ok"))
          end,
          tools: [tool]
        )

      {:ok, msg} = Message.new(role: :user, content: "Search")
      assert {:ok, _} = OpenAIService.chat(provider, [msg])
    end

    test "handles tool_call with invalid JSON in arguments" do
      response = %{
        "choices" => [
          %{
            "message" => %{
              "tool_calls" => [
                %{"function" => %{"name" => "test", "arguments" => "not-json"}}
              ]
            }
          }
        ]
      }

      provider = build_provider(fn conn -> Req.Test.json(conn, response) end)
      {:ok, msg} = Message.new(role: :user, content: "Test")

      assert {:tool_call, "test", %{"raw" => "not-json"}} =
               OpenAIService.chat(provider, [msg])
    end

    test "passes temperature and max_tokens in request" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          assert parsed["temperature"] == 0.5
          assert parsed["max_tokens"] == 100
          Req.Test.json(conn, success_response("Ok"))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")

      assert {:ok, _} =
               OpenAIService.chat(provider, [msg],
                 temperature: 0.5,
                 max_tokens: 100
               )
    end

    test "formats user message with attachments as multipart content" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [msg] = parsed["messages"]
          assert msg["role"] == "user"
          [image_part, text_part] = msg["content"]
          assert image_part["type"] == "image_url"
          assert String.starts_with?(image_part["image_url"]["url"], "data:image/png;base64,")
          assert text_part["type"] == "text"
          assert text_part["text"] == "Describe this"
          Req.Test.json(conn, success_response("An image"))
        end)

      {:ok, msg} =
        Message.new(
          role: :user,
          content: "Describe this",
          attachments: [%{data: "fake_png_data", mime_type: "image/png"}]
        )

      assert {:ok, _} = OpenAIService.chat(provider, [msg])
    end

    test "formats assistant message with tool_calls" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          assistant_msg = Enum.at(parsed["messages"], 1)
          assert assistant_msg["role"] == "assistant"
          [tc] = assistant_msg["tool_calls"]
          assert tc["type"] == "function"
          assert tc["function"]["name"] == "search"
          assert Jason.decode!(tc["function"]["arguments"]) == %{"q" => "elixir"}
          Req.Test.json(conn, success_response("Ok"))
        end)

      {:ok, user_msg} = Message.new(role: :user, content: "Search")

      {:ok, assistant_msg} =
        Message.new(
          role: :assistant,
          content: "",
          tool_calls: [%{"name" => "search", "args" => %{"q" => "elixir"}}]
        )

      assert {:ok, _} = OpenAIService.chat(provider, [user_msg, assistant_msg])
    end

    test "formats tool response message with tool_call_id" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          tool_msg = Enum.at(parsed["messages"], 1)
          assert tool_msg["role"] == "tool"
          assert tool_msg["content"] == "result data"
          assert tool_msg["tool_call_id"] == "search"
          Req.Test.json(conn, success_response("Ok"))
        end)

      {:ok, user_msg} = Message.new(role: :user, content: "Search")
      {:ok, tool_msg} = Message.new(role: :tool, content: "result data", tool_call_id: "search")

      assert {:ok, _} = OpenAIService.chat(provider, [user_msg, tool_msg])
    end
  end

  describe "chat/3 file_ref attachments" do
    test "formats image file_ref as image_file with file_id" do
      {:ok, ref} =
        ExAgent.FileRef.new(
          provider: :openai,
          file_id: "file-img123",
          mime_type: "image/png"
        )

      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [msg] = parsed["messages"]
          [file_part, text_part] = msg["content"]
          assert file_part["type"] == "image_file"
          assert file_part["image_file"]["file_id"] == "file-img123"
          assert text_part["type"] == "text"
          Req.Test.json(conn, success_response("An image"))
        end)

      {:ok, msg} =
        Message.new(
          role: :user,
          content: "Describe this",
          attachments: [%{file_ref: ref}]
        )

      assert {:ok, _} = OpenAIService.chat(provider, [msg])
    end

    test "formats non-image file_ref as file with file_id" do
      {:ok, ref} =
        ExAgent.FileRef.new(
          provider: :openai,
          file_id: "file-pdf456",
          mime_type: "application/pdf"
        )

      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [msg] = parsed["messages"]
          [file_part, _text_part] = msg["content"]
          assert file_part["type"] == "file"
          assert file_part["file"]["file_id"] == "file-pdf456"
          Req.Test.json(conn, success_response("A PDF"))
        end)

      {:ok, msg} =
        Message.new(
          role: :user,
          content: "Summarize",
          attachments: [%{file_ref: ref}]
        )

      assert {:ok, _} = OpenAIService.chat(provider, [msg])
    end

    test "mixes file_ref and inline attachments" do
      {:ok, ref} =
        ExAgent.FileRef.new(
          provider: :openai,
          file_id: "file-mix789",
          mime_type: "application/pdf"
        )

      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [msg] = parsed["messages"]
          [ref_part, inline_part, text_part] = msg["content"]
          assert ref_part["type"] == "file"
          assert ref_part["file"]["file_id"] == "file-mix789"
          assert inline_part["type"] == "image_url"
          assert text_part["type"] == "text"
          Req.Test.json(conn, success_response("Both files"))
        end)

      {:ok, msg} =
        Message.new(
          role: :user,
          content: "Compare",
          attachments: [
            %{file_ref: ref},
            %{data: "fake_png", mime_type: "image/png"}
          ]
        )

      assert {:ok, _} = OpenAIService.chat(provider, [msg])
    end

    test "formats inline non-image attachment as file with filename and file_data" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [msg] = parsed["messages"]
          [file_part, text_part] = msg["content"]
          assert file_part["type"] == "file"
          assert file_part["file"]["filename"] == "upload"
          assert String.starts_with?(file_part["file"]["file_data"], "data:application/pdf;base64,")
          refute Map.has_key?(file_part["file"], "file_id")
          assert text_part["type"] == "text"
          Req.Test.json(conn, success_response("A PDF"))
        end)

      {:ok, msg} =
        Message.new(
          role: :user,
          content: "Summarize this",
          attachments: [%{data: "fake_pdf_data", mime_type: "application/pdf"}]
        )

      assert {:ok, _} = OpenAIService.chat(provider, [msg])
    end

    test "formats inline non-image attachment with preserved filename" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [msg] = parsed["messages"]
          [file_part, _] = msg["content"]
          assert file_part["file"]["filename"] == "report.pdf"
          Req.Test.json(conn, success_response("Ok"))
        end)

      {:ok, msg} =
        Message.new(
          role: :user,
          content: "Read this",
          attachments: [%{data: "pdf_bytes", mime_type: "application/pdf", filename: "report.pdf"}]
        )

      assert {:ok, _} = OpenAIService.chat(provider, [msg])
    end
  end

  describe "built_in_tools" do
    test "adds web_search_options to request body" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          assert parsed["web_search_options"] == %{}
          Req.Test.json(conn, success_response("Search results"))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Search")

      assert {:ok, _} =
               OpenAIService.chat(provider, [msg], built_in_tools: [:web_search])
    end

    test "adds web_search_options with user location" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          assert parsed["web_search_options"]["user_location"] == %{
                   "type" => "approximate",
                   "approximate" => %{"city" => "Tokyo"}
                 }
          Req.Test.json(conn, success_response("Local results"))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Search")

      assert {:ok, _} =
               OpenAIService.chat(provider, [msg],
                 built_in_tools: [%{web_search: %{"city" => "Tokyo"}}]
               )
    end

    test "ignores unknown built-in tools" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          refute Map.has_key?(parsed, "web_search_options")
          Req.Test.json(conn, success_response("Ok"))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")

      assert {:ok, _} =
               OpenAIService.chat(provider, [msg], built_in_tools: [:unknown_tool])
    end
  end
end
