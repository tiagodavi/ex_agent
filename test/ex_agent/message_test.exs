defmodule ExAgent.MessageTest do
  use ExUnit.Case, async: true

  alias ExAgent.Message

  doctest ExAgent.Message

  # Happy path tests
  describe "new/1 with valid attrs" do
    test "creates a user message with content" do
      assert {:ok, %Message{role: :user, content: "Hello"}} =
               Message.new(role: :user, content: "Hello")
    end

    test "creates an assistant message with tool_calls" do
      tool_calls = [%{"id" => "call_1", "function" => %{"name" => "search"}}]

      assert {:ok, %Message{role: :assistant, tool_calls: ^tool_calls}} =
               Message.new(role: :assistant, content: "Let me search", tool_calls: tool_calls)
    end

    test "creates a tool message with tool_call_id" do
      assert {:ok, %Message{role: :tool, tool_call_id: "call_1"}} =
               Message.new(role: :tool, content: "result", tool_call_id: "call_1")
    end
  end

  # Bad path tests
  describe "new/1 with invalid attrs" do
    test "returns error for missing role" do
      assert {:error, _reason} = Message.new(content: "Hello")
    end

    test "returns error for invalid role" do
      assert {:error, _reason} = Message.new(role: :invalid, content: "Hello")
    end

    test "returns error for missing content" do
      assert {:error, _reason} = Message.new(role: :user)
    end
  end

  # File ref attachment tests
  describe "new/1 with file_ref attachments" do
    test "accepts a file_ref attachment" do
      {:ok, ref} =
        ExAgent.FileRef.new(
          provider: :openai,
          file_id: "file-abc123",
          mime_type: "application/pdf"
        )

      assert {:ok, %Message{attachments: [%{file_ref: ^ref}]}} =
               Message.new(role: :user, content: "Describe this", attachments: [%{file_ref: ref}])
    end

    test "accepts mixed inline and file_ref attachments" do
      {:ok, ref} =
        ExAgent.FileRef.new(
          provider: :gemini,
          file_uri: "https://example.com/files/abc",
          mime_type: "image/png"
        )

      attachments = [
        %{file_ref: ref},
        %{data: "inline data", mime_type: "text/plain"}
      ]

      assert {:ok, %Message{attachments: [%{file_ref: _}, %{data: _, mime_type: _}]}} =
               Message.new(role: :user, content: "Compare", attachments: attachments)
    end

    test "rejects invalid file_ref attachment" do
      assert {:error, _} =
               Message.new(
                 role: :user,
                 content: "Bad",
                 attachments: [%{file_ref: "not a struct"}]
               )
    end
  end

  # Edge case tests
  describe "new/1 edge cases" do
    test "creates a system message" do
      assert {:ok, %Message{role: :system, content: "You are helpful"}} =
               Message.new(role: :system, content: "You are helpful")
    end

    test "defaults metadata to empty map" do
      assert {:ok, %Message{metadata: %{}}} = Message.new(role: :user, content: "Hi")
    end

    test "accepts custom metadata" do
      assert {:ok, %Message{metadata: %{source: "test"}}} =
               Message.new(role: :user, content: "Hi", metadata: %{source: "test"})
    end

    test "path attachment preserves filename after resolution" do
      path = Path.join(System.tmp_dir!(), "test_doc.pdf")
      File.write!(path, "fake pdf content")

      {:ok, msg} =
        Message.new(
          role: :user,
          content: "Read this",
          attachments: [%{path: path, mime_type: "application/pdf"}]
        )

      [attachment] = msg.attachments
      assert attachment.filename == "test_doc.pdf"
      assert attachment.mime_type == "application/pdf"
      assert attachment.data == "fake pdf content"

      File.rm!(path)
    end
  end
end
