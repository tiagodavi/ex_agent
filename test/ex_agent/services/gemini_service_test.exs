defmodule ExAgent.Services.GeminiServiceTest do
  use ExUnit.Case, async: true

  alias ExAgent.Services.GeminiService
  alias ExAgent.Providers.Gemini
  alias ExAgent.{Message, Tool}

  defp success_response(text) do
    %{
      "candidates" => [
        %{"content" => %{"parts" => [%{"text" => text}], "role" => "model"}}
      ]
    }
  end

  defp tool_call_response(name, args) do
    %{
      "candidates" => [
        %{
          "content" => %{
            "parts" => [%{"functionCall" => %{"name" => name, "args" => args}}],
            "role" => "model"
          }
        }
      ]
    }
  end

  defp build_provider(plug_fn, opts \\ []) do
    req = Req.new(plug: plug_fn)

    %Gemini{
      api_key: "AIza-test",
      model: opts[:model] || "gemini-2.0-flash",
      base_url: "https://generativelanguage.googleapis.com/v1beta",
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
               GeminiService.chat(provider, [msg])
    end

    test "sends system_instruction when system_prompt is provided" do
      provider =
        build_provider(
          fn conn ->
            {:ok, body, conn} = Plug.Conn.read_body(conn)
            parsed = Jason.decode!(body)
            assert parsed["system_instruction"]["parts"] == [%{"text" => "Be brief"}]
            Req.Test.json(conn, success_response("Ok"))
          end,
          system_prompt: "Be brief"
        )

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:ok, _} = GeminiService.chat(provider, [msg])
    end

    test "returns tool_call when LLM requests function call" do
      provider =
        build_provider(fn conn ->
          Req.Test.json(conn, tool_call_response("search", %{"q" => "elixir"}))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Search")

      assert {:tool_call, "search", %{"q" => "elixir"}} =
               GeminiService.chat(provider, [msg])
    end
  end

  # Bad path tests
  describe "chat/3 errors" do
    test "returns error for non-200 status" do
      provider =
        build_provider(fn conn ->
          conn |> Plug.Conn.send_resp(403, Jason.encode!(%{"error" => "forbidden"}))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:error, {403, _}} = GeminiService.chat(provider, [msg])
    end

    test "returns error for unexpected response format" do
      provider =
        build_provider(fn conn ->
          Req.Test.json(conn, %{"bad" => "format"})
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")

      assert {:error, {:unexpected_response, _}} =
               GeminiService.chat(provider, [msg])
    end

    test "returns error for server errors" do
      provider =
        build_provider(fn conn ->
          conn |> Plug.Conn.send_resp(500, Jason.encode!(%{"error" => "internal"}))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:error, {500, _}} = GeminiService.chat(provider, [msg])
    end
  end

  # Edge case tests
  describe "chat/3 edge cases" do
    test "formats messages as Gemini contents with parts" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [content] = parsed["contents"]
          assert content["role"] == "user"
          assert content["parts"] == [%{"text" => "Hello"}]
          Req.Test.json(conn, success_response("Hi"))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Hello")
      assert {:ok, _} = GeminiService.chat(provider, [msg])
    end

    test "sends tools as functionDeclarations" do
      {:ok, tool} =
        Tool.new(
          name: "calc",
          description: "Calculate",
          parameters: %{"type" => "object"},
          function: & &1
        )

      provider =
        build_provider(
          fn conn ->
            {:ok, body, conn} = Plug.Conn.read_body(conn)
            parsed = Jason.decode!(body)
            [tools_group] = parsed["tools"]
            [decl] = tools_group["functionDeclarations"]
            assert decl["name"] == "calc"
            Req.Test.json(conn, success_response("Ok"))
          end,
          tools: [tool]
        )

      {:ok, msg} = Message.new(role: :user, content: "Calc")
      assert {:ok, _} = GeminiService.chat(provider, [msg])
    end

    test "uses correct URL pattern with model name" do
      provider =
        build_provider(
          fn conn ->
            assert conn.request_path == "/models/gemini-1.5-pro:generateContent"
            Req.Test.json(conn, success_response("Hi"))
          end,
          model: "gemini-1.5-pro"
        )

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:ok, _} = GeminiService.chat(provider, [msg])
    end

    test "formats tool response with role user and functionResponse" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          tool_content = Enum.at(parsed["contents"], 1)
          assert tool_content["role"] == "user"
          [part] = tool_content["parts"]
          assert part["functionResponse"]["name"] == "search"
          assert part["functionResponse"]["response"]["result"] == "found it"
          Req.Test.json(conn, success_response("Ok"))
        end)

      {:ok, user_msg} = Message.new(role: :user, content: "Search")
      {:ok, tool_msg} = Message.new(role: :tool, content: "found it", tool_call_id: "search")
      assert {:ok, _} = GeminiService.chat(provider, [user_msg, tool_msg])
    end

    test "formats assistant message with tool_calls as functionCall" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          assistant_content = Enum.at(parsed["contents"], 1)
          assert assistant_content["role"] == "model"
          [part] = assistant_content["parts"]
          assert part["functionCall"]["name"] == "search"
          assert part["functionCall"]["args"] == %{"q" => "elixir"}
          Req.Test.json(conn, success_response("Ok"))
        end)

      {:ok, user_msg} = Message.new(role: :user, content: "Search")

      {:ok, assistant_msg} =
        Message.new(
          role: :assistant,
          content: "",
          tool_calls: [%{"name" => "search", "args" => %{"q" => "elixir"}}]
        )

      assert {:ok, _} =
               GeminiService.chat(provider, [user_msg, assistant_msg])
    end

    test "formats user message with attachments as inline_data" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [content] = parsed["contents"]
          assert content["role"] == "user"
          [inline_part, text_part] = content["parts"]
          assert inline_part["inline_data"]["mime_type"] == "image/png"
          assert text_part["text"] == "Describe this"
          Req.Test.json(conn, success_response("An image"))
        end)

      {:ok, msg} =
        Message.new(
          role: :user,
          content: "Describe this",
          attachments: [%{data: "fake_png_data", mime_type: "image/png"}]
        )

      assert {:ok, _} = GeminiService.chat(provider, [msg])
    end
  end

  describe "chat/3 file_ref attachments" do
    test "formats file_ref as file_data with file_uri" do
      {:ok, ref} =
        ExAgent.FileRef.new(
          provider: :gemini,
          file_uri: "https://generativelanguage.googleapis.com/v1beta/files/abc123",
          mime_type: "application/pdf"
        )

      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [content] = parsed["contents"]
          [file_part, text_part] = content["parts"]

          assert file_part["file_data"]["file_uri"] ==
                   "https://generativelanguage.googleapis.com/v1beta/files/abc123"

          assert file_part["file_data"]["mime_type"] == "application/pdf"
          assert text_part["text"] == "Summarize"
          Req.Test.json(conn, success_response("A PDF"))
        end)

      {:ok, msg} =
        Message.new(
          role: :user,
          content: "Summarize",
          attachments: [%{file_ref: ref}]
        )

      assert {:ok, _} = GeminiService.chat(provider, [msg])
    end

    test "formats image file_ref as file_data" do
      {:ok, ref} =
        ExAgent.FileRef.new(
          provider: :gemini,
          file_uri: "https://example.com/files/img456",
          mime_type: "image/jpeg"
        )

      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [content] = parsed["contents"]
          [file_part, _text_part] = content["parts"]
          assert file_part["file_data"]["file_uri"] == "https://example.com/files/img456"
          assert file_part["file_data"]["mime_type"] == "image/jpeg"
          Req.Test.json(conn, success_response("An image"))
        end)

      {:ok, msg} =
        Message.new(
          role: :user,
          content: "Describe",
          attachments: [%{file_ref: ref}]
        )

      assert {:ok, _} = GeminiService.chat(provider, [msg])
    end

    test "mixes file_ref and inline attachments" do
      {:ok, ref} =
        ExAgent.FileRef.new(
          provider: :gemini,
          file_uri: "https://example.com/files/mix789",
          mime_type: "application/pdf"
        )

      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          [content] = parsed["contents"]
          [ref_part, inline_part, text_part] = content["parts"]
          assert ref_part["file_data"]["file_uri"] == "https://example.com/files/mix789"
          assert inline_part["inline_data"]["mime_type"] == "image/png"
          assert text_part["text"] == "Compare"
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

      assert {:ok, _} = GeminiService.chat(provider, [msg])
    end
  end

  describe "built_in_tools" do
    test "adds google_search to tools array" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          tools = parsed["tools"]
          assert Enum.any?(tools, &Map.has_key?(&1, "google_search"))
          Req.Test.json(conn, success_response("Search results"))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Search")

      assert {:ok, _} =
               GeminiService.chat(provider, [msg], built_in_tools: [:google_search])
    end

    test "adds code_execution to tools array" do
      provider =
        build_provider(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          parsed = Jason.decode!(body)
          tools = parsed["tools"]
          assert Enum.any?(tools, &Map.has_key?(&1, "code_execution"))
          Req.Test.json(conn, success_response("Code result"))
        end)

      {:ok, msg} = Message.new(role: :user, content: "Run code")

      assert {:ok, _} =
               GeminiService.chat(provider, [msg], built_in_tools: [:code_execution])
    end

    test "combines function declarations with built-in tools" do
      {:ok, tool} =
        Tool.new(
          name: "calc",
          description: "Calculate",
          parameters: %{"type" => "object"},
          function: & &1
        )

      provider =
        build_provider(
          fn conn ->
            {:ok, body, conn} = Plug.Conn.read_body(conn)
            parsed = Jason.decode!(body)
            tools = parsed["tools"]
            assert length(tools) == 2
            assert Enum.any?(tools, &Map.has_key?(&1, "functionDeclarations"))
            assert Enum.any?(tools, &Map.has_key?(&1, "google_search"))
            Req.Test.json(conn, success_response("Ok"))
          end,
          tools: [tool]
        )

      {:ok, msg} = Message.new(role: :user, content: "Calc")

      assert {:ok, _} =
               GeminiService.chat(provider, [msg], built_in_tools: [:google_search])
    end
  end
end
