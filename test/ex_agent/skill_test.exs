defmodule ExAgent.SkillTest do
  use ExUnit.Case, async: true

  alias ExAgent.{Skill, Context, Message, Tool}

  doctest ExAgent.Skill

  # Happy path tests
  describe "new/1 with valid attrs" do
    test "creates a skill with name and system_prompt" do
      assert {:ok, %Skill{name: "sql_expert", system_prompt: "You are a SQL expert"}} =
               Skill.new(name: "sql_expert", system_prompt: "You are a SQL expert")
    end

    test "creates a skill with tools" do
      {:ok, tool} =
        Tool.new(name: "query", description: "Run SQL", parameters: %{}, function: & &1)

      {:ok, skill} =
        Skill.new(name: "sql_expert", system_prompt: "SQL expert", tools: [tool])

      assert length(skill.tools) == 1
    end

    test "creates a skill with activation function" do
      activation_fn = fn %Context{} -> true end

      {:ok, skill} =
        Skill.new(
          name: "sql_expert",
          system_prompt: "SQL expert",
          activation_fn: activation_fn
        )

      assert is_function(skill.activation_fn, 1)
    end
  end

  # Bad path tests
  describe "new/1 with invalid attrs" do
    test "returns error for missing name" do
      assert {:error, _reason} = Skill.new(system_prompt: "Expert")
    end

    test "returns error for missing system_prompt" do
      assert {:error, _reason} = Skill.new(name: "expert")
    end

    test "returns error for invalid activation_fn arity" do
      assert {:error, _reason} =
               Skill.new(
                 name: "expert",
                 system_prompt: "Expert",
                 activation_fn: fn _a, _b -> true end
               )
    end
  end

  # Edge case tests
  describe "edge cases" do
    test "tools default to empty list" do
      {:ok, skill} = Skill.new(name: "expert", system_prompt: "Expert")
      assert skill.tools == []
    end

    test "activation_fn defaults to nil" do
      {:ok, skill} = Skill.new(name: "expert", system_prompt: "Expert")
      assert skill.activation_fn == nil
    end

    test "activation function receives context and returns boolean" do
      {:ok, msg} = Message.new(role: :user, content: "SELECT * FROM users")
      context = Context.new() |> Context.add_message(msg)

      activation_fn = fn ctx ->
        ctx.messages
        |> Enum.any?(fn m -> String.contains?(m.content, "SELECT") end)
      end

      {:ok, skill} =
        Skill.new(name: "sql", system_prompt: "SQL", activation_fn: activation_fn)

      assert skill.activation_fn.(context) == true
    end
  end
end
