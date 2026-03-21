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
  end
end
