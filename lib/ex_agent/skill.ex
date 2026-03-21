defmodule ExAgent.Skill do
  @moduledoc """
  Loadable skill definition for progressive disclosure pattern.

  A skill represents a specialized persona with its own system prompt,
  tools, and an optional activation function that determines when
  the skill should be dynamically loaded based on conversation context.
  """

  alias ExAgent.Tool

  @type t :: %__MODULE__{
          name: String.t(),
          system_prompt: String.t(),
          tools: [Tool.t()],
          activation_fn: (ExAgent.Context.t() -> boolean()) | nil
        }

  @enforce_keys [:name, :system_prompt]
  defstruct [:name, :system_prompt, :activation_fn, tools: []]

  @doc """
  Creates a new skill with validated attributes.

  ## Examples

      iex> {:ok, skill} = ExAgent.Skill.new(name: "expert", system_prompt: "You are an expert")
      iex> skill.name
      "expert"

      iex> ExAgent.Skill.new(system_prompt: "Expert")
      {:error, "name is required"}

      iex> ExAgent.Skill.new(name: "expert")
      {:error, "system_prompt is required"}
  """
  @spec new(keyword()) :: {:ok, t()} | {:error, String.t()}
  def new(attrs) when is_list(attrs) do
    with {:ok, name} <- validate_required_string(attrs[:name], "name"),
         {:ok, system_prompt} <- validate_required_string(attrs[:system_prompt], "system_prompt"),
         {:ok, activation_fn} <- validate_activation_fn(attrs[:activation_fn]) do
      {:ok,
       %__MODULE__{
         name: name,
         system_prompt: system_prompt,
         tools: attrs[:tools] || [],
         activation_fn: activation_fn
       }}
    end
  end

  @spec validate_required_string(any(), String.t()) :: {:ok, String.t()} | {:error, String.t()}
  defp validate_required_string(nil, field), do: {:error, "#{field} is required"}
  defp validate_required_string(value, _field) when is_binary(value), do: {:ok, value}
  defp validate_required_string(_, field), do: {:error, "#{field} must be a string"}

  @spec validate_activation_fn(any()) :: {:ok, function() | nil} | {:error, String.t()}
  defp validate_activation_fn(nil), do: {:ok, nil}
  defp validate_activation_fn(fun) when is_function(fun, 1), do: {:ok, fun}

  defp validate_activation_fn(_),
    do: {:error, "activation_fn must be a function with arity 1"}
end
