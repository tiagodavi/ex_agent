defmodule ExAgent.Tool do
  @moduledoc """
  Normalized tool definition for LLM function-calling.

  Represents a tool that an LLM can invoke, including its name,
  description, JSON Schema parameters, and the actual function to execute.
  """

  @type t :: %__MODULE__{
          name: String.t(),
          description: String.t(),
          parameters: map(),
          function: (map() -> any())
        }

  @enforce_keys [:name, :description, :function]
  defstruct [:name, :description, :function, parameters: %{}]

  @doc """
  Creates a new tool with validated attributes.

  ## Examples

      iex> {:ok, tool} = ExAgent.Tool.new(name: "search", description: "Search the web", parameters: %{}, function: fn _ -> :ok end)
      iex> tool.name
      "search"

      iex> ExAgent.Tool.new(description: "Search", parameters: %{}, function: fn _ -> :ok end)
      {:error, "name is required"}

      iex> ExAgent.Tool.new(name: "search", parameters: %{}, function: fn _ -> :ok end)
      {:error, "description is required"}
  """
  @spec new(keyword()) :: {:ok, t()} | {:error, String.t()}
  def new(attrs) when is_list(attrs) do
    with {:ok, name} <- validate_required_string(attrs[:name], "name"),
         {:ok, description} <- validate_required_string(attrs[:description], "description"),
         {:ok, function} <- validate_function(attrs[:function]) do
      {:ok,
       %__MODULE__{
         name: name,
         description: description,
         parameters: attrs[:parameters] || %{},
         function: function
       }}
    end
  end

  @spec validate_required_string(any(), String.t()) :: {:ok, String.t()} | {:error, String.t()}
  defp validate_required_string(nil, field), do: {:error, "#{field} is required"}
  defp validate_required_string(value, _field) when is_binary(value), do: {:ok, value}
  defp validate_required_string(_, field), do: {:error, "#{field} must be a string"}

  @spec validate_function(any()) :: {:ok, function()} | {:error, String.t()}
  defp validate_function(nil), do: {:error, "function is required"}
  defp validate_function(fun) when is_function(fun, 1), do: {:ok, fun}
  defp validate_function(_), do: {:error, "function must be a function with arity 1"}
end
