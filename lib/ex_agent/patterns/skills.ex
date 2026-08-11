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

  Providers with no `:system_prompt` field keep the skill's tools; the prompt
  simply has nowhere to go.
  """
  @spec apply_skill(map(), Skill.t()) :: map()
  def apply_skill(state, %Skill{} = skill) do
    %{state | provider: put_prompt(state.provider, skill.system_prompt), active_skill: skill}
  end

  @doc """
  Deactivates the active skill, restoring the agent's own system prompt.

  Skills are evaluated before every turn, so one that stops matching has to be
  undone — otherwise the first activation would overwrite the agent's persona
  permanently.
  """
  @spec clear_skill(map()) :: map()
  def clear_skill(%{active_skill: nil} = state), do: state

  def clear_skill(state) do
    %{
      state
      | provider: put_prompt(state.provider, Map.get(state, :base_system_prompt)),
        active_skill: nil
    }
  end

  @spec put_prompt(struct(), String.t() | nil) :: struct()
  defp put_prompt(provider, prompt) do
    if Map.has_key?(provider, :system_prompt),
      do: %{provider | system_prompt: prompt},
      else: provider
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
