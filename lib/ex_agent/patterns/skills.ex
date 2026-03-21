defmodule ExAgent.Patterns.Skills do
  @moduledoc """
  Progressive disclosure pattern.

  Allows a single agent to dynamically load specialized system prompts
  and tool sets based on conversation context. Skills are evaluated
  before each LLM call and the matching skill's persona is applied.
  """

  alias ExAgent.{Context, Skill}

  @doc """
  Evaluates all skills against the current context.

  Returns the first skill whose `activation_fn` returns `true`,
  or `nil` if no skill matches or has an activation function.
  """
  @spec evaluate_skills([Skill.t()], Context.t()) :: Skill.t() | nil
  def evaluate_skills(skills, %Context{} = context) do
    Enum.find(skills, fn
      %Skill{activation_fn: nil} -> false
      %Skill{activation_fn: fun} -> fun.(context)
    end)
  end

  @doc """
  Applies a skill to the agent state by updating the active_skill
  and injecting the skill's system prompt into the provider.
  """
  @spec apply_skill(map(), Skill.t()) :: map()
  def apply_skill(state, %Skill{} = skill) do
    provider = %{state.provider | system_prompt: skill.system_prompt}
    %{state | provider: provider, active_skill: skill}
  end

  @doc """
  Returns the effective tools: base tools plus active skill tools.
  """
  @spec effective_tools(map()) :: [ExAgent.Tool.t()]
  def effective_tools(%{tools: tools, active_skill: nil}), do: tools

  def effective_tools(%{tools: tools, active_skill: %Skill{tools: skill_tools}}) do
    tools ++ skill_tools
  end
end
