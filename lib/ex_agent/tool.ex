defmodule ExAgent.Tool do
  @moduledoc """
  Normalized tool definition for LLM function-calling.

  Represents a tool that an LLM can invoke, including its name,
  description, JSON Schema parameters, and the actual function to execute.

  `:parameters` takes either a raw JSON Schema map or an `Ecto` embedded schema
  module. With a module, the JSON Schema is generated for you and the arguments
  are cast before your function runs, so it receives a struct:

      defmodule OrderQuery do
        use Ecto.Schema

        @primary_key false
        embedded_schema do
          field :order_id, :string
        end
      end

      ExAgent.Tool.new(
        name: "order_status",
        description: "Look up the delivery status of an order",
        parameters: OrderQuery,
        function: fn %OrderQuery{order_id: id} -> Repo.get(Order, id) end
      )

  A typo in the function body is then a compile-time warning rather than a
  `KeyError` inside the agent's tool loop. Arguments the model gets wrong are fed
  back to it as a tool error, which is what lets it correct itself.
  """

  alias ExAgent.Schema

  @type t :: %__MODULE__{
          name: String.t(),
          description: String.t(),
          parameters: map(),
          schema: module() | nil,
          function: (map() | struct() -> any())
        }

  @enforce_keys [:name, :description, :function]
  defstruct [:name, :description, :function, :schema, parameters: %{}]

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
         {:ok, function} <- validate_function(attrs[:function]),
         {:ok, parameters, schema} <- validate_parameters(attrs[:parameters]) do
      {:ok,
       %__MODULE__{
         name: name,
         description: description,
         parameters: parameters,
         schema: schema,
         function: function
       }}
    end
  end

  @doc """
  Casts a model's raw arguments for `tool`.

  A tool declared with a raw JSON Schema map gets the arguments unchanged. One
  declared with an `Ecto` schema gets a struct, or `{:error, message}` phrased for
  the model when the arguments do not fit.
  """
  @spec cast_args(t(), map()) :: {:ok, map() | struct()} | {:error, String.t()}
  def cast_args(%__MODULE__{schema: nil}, args), do: {:ok, args}

  def cast_args(%__MODULE__{schema: schema}, args) do
    case Schema.cast(schema, args) do
      {:ok, struct} -> {:ok, struct}
      {:error, error} -> {:error, Exception.message(error)}
    end
  end

  # The `:gemini` dialect is the shape both providers accept in a *function
  # declaration*: Gemini rejects `additionalProperties` as an unknown field, and
  # OpenAI's non-strict function calling does not require it.
  @spec validate_parameters(term()) :: {:ok, map(), module() | nil} | {:error, String.t()}
  defp validate_parameters(nil), do: {:ok, %{}, nil}
  defp validate_parameters(parameters) when is_map(parameters), do: {:ok, parameters, nil}

  defp validate_parameters(module) when is_atom(module) do
    case Schema.to_json_schema(module, dialect: :gemini) do
      {:ok, json_schema} -> {:ok, json_schema, module}
      {:error, error} -> {:error, Exception.message(error)}
    end
  end

  defp validate_parameters(_parameters),
    do: {:error, "parameters must be a JSON Schema map or an Ecto schema module"}

  @spec validate_required_string(any(), String.t()) :: {:ok, String.t()} | {:error, String.t()}
  defp validate_required_string(nil, field), do: {:error, "#{field} is required"}
  defp validate_required_string(value, _field) when is_binary(value), do: {:ok, value}
  defp validate_required_string(_, field), do: {:error, "#{field} must be a string"}

  @spec validate_function(any()) :: {:ok, function()} | {:error, String.t()}
  defp validate_function(nil), do: {:error, "function is required"}
  defp validate_function(fun) when is_function(fun, 1), do: {:ok, fun}
  defp validate_function(_), do: {:error, "function must be a function with arity 1"}
end
