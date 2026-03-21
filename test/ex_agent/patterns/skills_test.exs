defmodule ExAgent.Patterns.SkillsTest do
  use ExUnit.Case, async: true

  alias ExAgent.Patterns.Skills
  alias ExAgent.{Context, Message, Skill, Tool}
  alias ExAgent.Providers.OpenAI

  # Happy path tests
  describe "evaluate_skills/2" do
    test "returns first matching skill" do
      {:ok, skill} =
        Skill.new(
          name: "sql",
          system_prompt: "SQL expert",
          activation_fn: fn _ctx -> true end
        )

      context = Context.new()
      assert %Skill{name: "sql"} = Skills.evaluate_skills([skill], context)
    end

    test "returns skill based on context content" do
      {:ok, skill} =
        Skill.new(
          name: "sql",
          system_prompt: "SQL expert",
          activation_fn: fn ctx ->
            Enum.any?(ctx.messages, &String.contains?(&1.content, "SELECT"))
          end
        )

      {:ok, msg} = Message.new(role: :user, content: "SELECT * FROM users")
      context = Context.new() |> Context.add_message(msg)
      assert %Skill{name: "sql"} = Skills.evaluate_skills([skill], context)
    end

    test "returns first matching skill when multiple match" do
      {:ok, skill1} = Skill.new(name: "first", system_prompt: "First", activation_fn: fn _ -> true end)
      {:ok, skill2} = Skill.new(name: "second", system_prompt: "Second", activation_fn: fn _ -> true end)

      assert %Skill{name: "first"} = Skills.evaluate_skills([skill1, skill2], Context.new())
    end
  end

  # Bad path tests
  describe "evaluate_skills/2 no match" do
    test "returns nil when no skills match" do
      {:ok, skill} =
        Skill.new(name: "sql", system_prompt: "SQL", activation_fn: fn _ -> false end)

      assert nil == Skills.evaluate_skills([skill], Context.new())
    end

    test "returns nil for empty skills list" do
      assert nil == Skills.evaluate_skills([], Context.new())
    end

    test "skips skills without activation_fn" do
      {:ok, skill} = Skill.new(name: "passive", system_prompt: "Passive")
      assert nil == Skills.evaluate_skills([skill], Context.new())
    end
  end

  # Edge case tests
  describe "apply_skill/2" do
    test "updates provider system_prompt and active_skill" do
      provider = %OpenAI{api_key: "test", req: nil}
      state = %{provider: provider, active_skill: nil}

      {:ok, skill} = Skill.new(name: "expert", system_prompt: "Expert mode")
      new_state = Skills.apply_skill(state, skill)

      assert new_state.provider.system_prompt == "Expert mode"
      assert new_state.active_skill.name == "expert"
    end

    test "effective_tools returns base tools when no active skill" do
      state = %{tools: [%Tool{name: "a", description: "A", parameters: %{}, function: & &1}], active_skill: nil}
      assert length(Skills.effective_tools(state)) == 1
    end

    test "effective_tools merges base tools with skill tools" do
      base_tool = %Tool{name: "a", description: "A", parameters: %{}, function: & &1}
      skill_tool = %Tool{name: "b", description: "B", parameters: %{}, function: & &1}

      {:ok, skill} = Skill.new(name: "expert", system_prompt: "Expert", tools: [skill_tool])
      state = %{tools: [base_tool], active_skill: skill}

      assert length(Skills.effective_tools(state)) == 2
    end
  end
end
