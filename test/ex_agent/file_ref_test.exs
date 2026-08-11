defmodule ExAgent.FileRefTest do
  use ExUnit.Case, async: true

  alias ExAgent.FileRef

  doctest ExAgent.FileRef

  # Happy path tests
  describe "new/1 with valid attrs" do
    test "creates an OpenAI file reference with file_id" do
      assert {:ok,
              %FileRef{provider: :openai, file_id: "file-abc123", mime_type: "application/pdf"}} =
               FileRef.new(
                 provider: :openai,
                 file_id: "file-abc123",
                 mime_type: "application/pdf"
               )
    end

    test "creates a Gemini file reference with file_uri" do
      assert {:ok,
              %FileRef{
                provider: :gemini,
                file_uri: "https://example.com/files/abc",
                mime_type: "image/png"
              }} =
               FileRef.new(
                 provider: :gemini,
                 file_uri: "https://example.com/files/abc",
                 mime_type: "image/png"
               )
    end

    test "accepts optional filename and expires_at" do
      expires = DateTime.add(DateTime.utc_now(), 48 * 3600, :second)

      assert {:ok, %FileRef{filename: "report.pdf", expires_at: ^expires}} =
               FileRef.new(
                 provider: :gemini,
                 file_uri: "https://example.com/files/abc",
                 mime_type: "application/pdf",
                 filename: "report.pdf",
                 expires_at: expires
               )
    end
  end

  # Bad path tests
  describe "new/1 with invalid attrs" do
    test "returns error for missing provider" do
      assert {:error, "provider is required"} =
               FileRef.new(file_id: "file-abc", mime_type: "text/plain")
    end

    test "returns error for OpenAI ref without file_id" do
      assert {:error, "OpenAI file references require :file_id"} =
               FileRef.new(provider: :openai, mime_type: "image/png")
    end

    test "returns error for Gemini ref without file_uri" do
      assert {:error, "Gemini file references require :file_uri"} =
               FileRef.new(provider: :gemini, mime_type: "image/png")
    end
  end

  # Edge case tests
  # A third-party provider implementing the optional upload/4 callback has to be
  # able to build the reference its own callback returns. The built-in services
  # construct %FileRef{} structs directly, so a closed provider list here never
  # protected them — it only walled out everyone else.
  describe "new/1 for custom providers" do
    test "accepts a provider this library has never heard of" do
      assert {:ok, ref} =
               FileRef.new(provider: :anthropic, file_id: "file-abc", mime_type: "text/plain")

      assert ref.provider == :anthropic
      assert ref.file_id == "file-abc"
    end

    test "accepts a custom provider identified by URI instead of id" do
      assert {:ok, ref} =
               FileRef.new(
                 provider: :my_llm,
                 file_uri: "https://x.test/f/1",
                 mime_type: "text/csv"
               )

      assert ref.file_uri == "https://x.test/f/1"
    end

    test "still requires the reference to actually reference something" do
      assert {:error, message} = FileRef.new(provider: :anthropic, mime_type: "text/plain")

      assert message =~ ":file_id"
      assert message =~ ":file_uri"
    end

    test "keeps enforcing the built-in providers' own field rules" do
      # OpenAI and Gemini services pattern-match on these fields, so a ref
      # carrying the wrong one would fail later and less clearly.
      assert {:error, "OpenAI file references require :file_id"} =
               FileRef.new(provider: :openai, file_uri: "https://x", mime_type: "text/plain")

      assert {:error, "Gemini file references require :file_uri"} =
               FileRef.new(provider: :gemini, file_id: "file-abc", mime_type: "text/plain")
    end
  end

  describe "new/1 edge cases" do
    test "returns error for missing mime_type" do
      assert {:error, "mime_type is required"} =
               FileRef.new(provider: :openai, file_id: "file-abc")
    end

    test "returns error for non-string mime_type" do
      assert {:error, "mime_type must be a string"} =
               FileRef.new(provider: :openai, file_id: "file-abc", mime_type: 123)
    end
  end

  describe "expired?/1" do
    test "returns false when expires_at is nil" do
      {:ok, ref} = FileRef.new(provider: :openai, file_id: "f-1", mime_type: "text/plain")
      refute FileRef.expired?(ref)
    end

    test "returns true when expires_at is in the past" do
      past = DateTime.add(DateTime.utc_now(), -3600, :second)

      {:ok, ref} =
        FileRef.new(
          provider: :gemini,
          file_uri: "https://example.com/files/abc",
          mime_type: "text/plain",
          expires_at: past
        )

      assert FileRef.expired?(ref)
    end

    test "returns false when expires_at is in the future" do
      future = DateTime.add(DateTime.utc_now(), 3600, :second)

      {:ok, ref} =
        FileRef.new(
          provider: :gemini,
          file_uri: "https://example.com/files/abc",
          mime_type: "text/plain",
          expires_at: future
        )

      refute FileRef.expired?(ref)
    end
  end
end
