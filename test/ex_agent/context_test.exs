defmodule ExAgent.ContextTest do
  use ExUnit.Case, async: true

  alias ExAgent.{Context, Message}

  doctest ExAgent.Context

  # Happy path tests
  describe "new/0 and new/1" do
    test "creates an empty context" do
      context = Context.new()
      assert context.messages == []
      assert context.metadata == %{}
    end

    test "creates a context with initial messages" do
      {:ok, msg} = Message.new(role: :user, content: "Hello")
      context = Context.new(messages: [msg])
      assert length(context.messages) == 1
    end

    test "creates a context with metadata" do
      context = Context.new(metadata: %{session_id: "abc123"})
      assert context.metadata == %{session_id: "abc123"}
    end
  end

  # Happy path - add_message
  describe "add_message/2" do
    test "appends a message to the context" do
      {:ok, msg} = Message.new(role: :user, content: "Hello")
      context = Context.new() |> Context.add_message(msg)
      assert length(context.messages) == 1
      assert hd(context.messages).content == "Hello"
    end

    test "preserves message order" do
      {:ok, msg1} = Message.new(role: :user, content: "First")
      {:ok, msg2} = Message.new(role: :assistant, content: "Second")

      context =
        Context.new()
        |> Context.add_message(msg1)
        |> Context.add_message(msg2)

      assert Enum.map(context.messages, & &1.content) == ["First", "Second"]
    end

    test "appends multiple messages in sequence" do
      {:ok, msg1} = Message.new(role: :user, content: "A")
      {:ok, msg2} = Message.new(role: :assistant, content: "B")
      {:ok, msg3} = Message.new(role: :user, content: "C")

      context =
        Context.new()
        |> Context.add_message(msg1)
        |> Context.add_message(msg2)
        |> Context.add_message(msg3)

      assert length(context.messages) == 3
    end
  end

  # Bad path tests
  describe "get_last_assistant_message/1" do
    test "returns nil when no messages exist" do
      assert Context.get_last_assistant_message(Context.new()) == nil
    end

    test "returns nil when no assistant messages exist" do
      {:ok, msg} = Message.new(role: :user, content: "Hello")
      context = Context.new() |> Context.add_message(msg)
      assert Context.get_last_assistant_message(context) == nil
    end

    test "returns the last assistant message" do
      {:ok, msg1} = Message.new(role: :assistant, content: "First reply")
      {:ok, msg2} = Message.new(role: :user, content: "Follow up")
      {:ok, msg3} = Message.new(role: :assistant, content: "Second reply")

      context =
        Context.new()
        |> Context.add_message(msg1)
        |> Context.add_message(msg2)
        |> Context.add_message(msg3)

      assert Context.get_last_assistant_message(context).content == "Second reply"
    end
  end

  # Edge case tests
  describe "edge cases" do
    test "parent_ref defaults to nil" do
      assert Context.new().parent_ref == nil
    end

    test "accepts a parent_ref" do
      ref = make_ref()
      context = Context.new(parent_ref: ref)
      assert context.parent_ref == ref
    end

    test "metadata is preserved when adding messages" do
      context = Context.new(metadata: %{key: "value"})
      {:ok, msg} = Message.new(role: :user, content: "Hello")
      updated = Context.add_message(context, msg)
      assert updated.metadata == %{key: "value"}
    end
  end
end
